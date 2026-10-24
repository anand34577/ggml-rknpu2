// RKNPU2 backend for Rockchip NPUs (RK3588, RK3576, RK3566/RK3568, RK3562)
//
// Offloads large matrix multiplications with static weights to the NPU through the
// rknn matmul API in librknnrt.so. Everything else stays on the CPU backend, the
// scheduler splits the graph for us. Same shape as the BLAS backend.
//
// The NPU computes C(M,N) = A(M,K) x B(K,N) in fp16 with fp32 accumulate.
// ggml mul_mat maps as: A = src1 (activations), B = src0 transposed (weights), C = dst.
// Weights are transposed, converted to the chip native layout once and kept in NPU
// memory. Quantized weights are dequantized to fp16 on the way in.
//
// The NPU has hard shape limits (K and N alignment, K max) that differ per chip, so
// supports_op only claims what the chip can do. Anything the runtime still rejects at
// setup time is remembered and never offered again, and a matmul that fails mid-flight
// falls back to a plain CPU loop, so a wrong guess costs speed but never correctness.

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-rknpu2.h"
#include "ggml-backend-impl.h"

#include "rknn_api.h"
#include "rknn_matmul_api.h"

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

// NPU limits per chip. K/N alignment is in elements and comes from the B native layout
// in rknn_matmul_api.h: fp16 B is (N/n_align, K/k_align, n_align, k_align).
//
// k_max is the largest K we ever hand to one rknn_matmul context/conversion call. The
// API's hard create()/set_io_mem() limit is K<=10240 on every chip, but
// rknn_B_normal_layout_to_native_layout silently switches to a segmented "T native
// layout" once K exceeds a smaller, chip-specific threshold (documented in
// rknn_matmul_api.h): 8192 on RK3588, 4096 on RK3576. A context built for a flat
// native buffer does not know about that segmented layout, so a chunk landing between
// the threshold and 10240 would convert to the wrong buffer shape and silently produce
// wrong matmul results. k_max is capped at that threshold (not just the 10240 create()
// limit) so plan_chunks() never emits a chunk that can trigger it. RK3566/3568/RK3562
// have no documented split threshold, so they keep the plain 10240 cap.
//
// n_cores is the number of NPU cores. The header says the matmul core mask is RK3588 only,
// so probe() tests pinning on the real chip and falls back to parallel unpinned contexts.
struct ggml_rknpu2_soc {
    const char * name;
    int64_t      k_align;
    int64_t      n_align;
    int64_t      k_max;
    int          n_cores;
};

static const ggml_rknpu2_soc ggml_rknpu2_socs[] = {
    { "rk3588", 32, 16, 8192,  3 },
    { "rk3576", 32, 16, 4096,  2 },
    { "rk3566", 16,  8, 10240, 1 },
    { "rk3568", 16,  8, 10240, 1 },
    { "rk3562", 32,  8, 10240, 1 },
};

// conservative default: the strictest alignment and the strictest known B-layout
// split threshold of any supported chip, in case an undetected chip has one too
static const ggml_rknpu2_soc ggml_rknpu2_soc_unknown = { "unknown", 32, 16, 4096, 1 };

static ggml_rknpu2_soc ggml_rknpu2_detect_soc(void) {
    std::string id;

    if (const char * env = getenv("GGML_RKNPU2_SOC")) {
        id = env;
    } else {
        // the compatible property holds NUL separated strings like "rockchip,rk3588"
        const char * paths[] = { "/proc/device-tree/compatible", "/sys/firmware/devicetree/base/compatible" };
        for (const char * path : paths) {
            FILE * f = fopen(path, "rb");
            if (f == nullptr) {
                continue;
            }
            char   buf[512];
            size_t n = fread(buf, 1, sizeof(buf) - 1, f);
            fclose(f);
            for (size_t i = 0; i < n; i++) {
                if (buf[i] == '\0') {
                    buf[i] = ' ';
                }
            }
            buf[n] = '\0';
            id = buf;
            break;
        }
    }

    for (char & c : id) {
        c = (char) tolower((unsigned char) c);
    }

    for (const ggml_rknpu2_soc & soc : ggml_rknpu2_socs) {
        if (id.find(soc.name) != std::string::npos) {
            return soc;
        }
    }

    return ggml_rknpu2_soc_unknown;
}

static int64_t ggml_rknpu2_env_i64(const char * name, int64_t def) {
    const char * env = getenv(name);
    if (env == nullptr) {
        return def;
    }
    char *  end = nullptr;
    int64_t v   = strtoll(env, &end, 10);
    return (end != env && v > 0) ? v : def;
}

// one prepared rknn matmul context, fixed shape (m_tile, K, N)
struct ggml_rknpu2_kernel {
    rknn_matmul_ctx     ctx = 0;
    rknn_matmul_info    info;
    rknn_matmul_io_attr io_attr;
    int64_t             m_tile = 0;
    rknn_tensor_mem *   A = nullptr;
    rknn_tensor_mem *   C = nullptr;
    const void *        bound_B = nullptr;
};

// NPU resident native layout copy of one weight, indexed [column slice][K chunk]
struct ggml_rknpu2_weight {
    std::vector<std::vector<rknn_tensor_mem *>> B;
    std::vector<std::vector<rknn_matmul_ctx>>   ctxs;
    uint64_t                                    fingerprint = 0;
    size_t                                      bytes = 0;
};

// one column range of the output, run on its own NPU core
struct ggml_rknpu2_slice {
    int64_t n0;
    int64_t nc;
    int     core;
};

// Workers for the extra NPU cores. Started once: a matmul runs for a few ms, so creating a
// thread per call costs a large part of what the extra core wins.
struct ggml_rknpu2_workers {
    std::vector<std::thread> threads;

    std::mutex              mutex;
    std::condition_variable cv_work;
    std::condition_variable cv_done;

    std::function<void(size_t)> fn;
    uint64_t                    gen      = 0;
    size_t                      n_active = 0;
    size_t                      pending  = 0;
    bool                        stop     = false;

    ~ggml_rknpu2_workers() { shutdown(); }

    void start(size_t n) {
        // gen survives a shutdown, new workers must not take an old job as new work
        uint64_t cur;
        {
            std::lock_guard<std::mutex> lock(mutex);
            cur = gen;
        }
        for (size_t i = 0; i < n; i++) {
            threads.emplace_back([this, i, cur] { loop(i, cur); });
        }
    }

    void loop(size_t idx, uint64_t seen) {
        for (;;) {
            std::unique_lock<std::mutex> lock(mutex);
            cv_work.wait(lock, [&] { return stop || gen != seen; });
            if (stop) {
                return;
            }
            seen = gen;
            if (idx >= n_active) {
                continue;
            }
            std::function<void(size_t)> job = fn;
            lock.unlock();

            job(idx + 1);

            lock.lock();
            if (--pending == 0) {
                cv_done.notify_one();
            }
        }
    }

    // run job(0) here and job(1..n) on the workers, return when all are done
    void run(size_t n, const std::function<void(size_t)> & job) {
        if (n > threads.size()) {
            n = threads.size();
        }
        if (n > 0) {
            std::lock_guard<std::mutex> lock(mutex);
            fn       = job;
            n_active = n;
            pending  = n;
            gen++;
            cv_work.notify_all();
        }

        job(0);

        if (n > 0) {
            std::unique_lock<std::mutex> lock(mutex);
            cv_done.wait(lock, [&] { return pending == 0; });
        }
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop = true;
            gen++;
        }
        cv_work.notify_all();
        for (std::thread & t : threads) {
            t.join();
        }
        threads.clear();
        stop     = false;
        fn       = nullptr;
        n_active = 0;
        pending  = 0;
    }
};

struct ggml_rknpu2_weight_key {
    const void * data;
    int64_t      k;
    int64_t      n;
    ggml_type    type;

    bool operator<(const ggml_rknpu2_weight_key & o) const {
        return std::tie(data, k, n, type) < std::tie(o.data, o.k, o.n, o.type);
    }
};

// (K chunk, N slice, core) - the core is part of the key because a context carries its
// own core mask, and two cores must not share one context
typedef std::tuple<int64_t, int64_t, int> ggml_rknpu2_shape;

struct ggml_rknpu2_state {
    std::mutex mutex;

    ggml_rknpu2_soc soc = ggml_rknpu2_soc_unknown;

    std::map<ggml_rknpu2_shape, ggml_rknpu2_kernel>      kernels;
    std::map<ggml_rknpu2_weight_key, ggml_rknpu2_weight> weights;
    std::set<ggml_rknpu2_shape>                          rejected;         // shapes the runtime refused
    std::set<ggml_rknpu2_weight_key>                     rejected_weights; // weights we could not put on the NPU
    std::map<ggml_rknpu2_weight_key, size_t>             reserved;         // accepted by supports_op, not uploaded yet

    size_t bytes_used     = 0;
    size_t bytes_reserved = 0;
    size_t bytes_cap      = 0; // 0 means only limited by allocation failures

    ggml_rknpu2_workers workers;

    int64_t m_min  = 0;
    int64_t m_max  = 0;
    int     n_cores   = 1;
    bool    pin_cores = false; // each slice context is pinned to its core, else the driver picks a free core
    bool    f32_weights = false;

    int n_backends   = 0;
    int availability = -1; // -1 unknown, 0 no NPU, 1 ok
    bool warned_fallback = false;
    bool warned_f32      = false;
};

static ggml_rknpu2_state & ggml_rknpu2_get_state(void) {
    static ggml_rknpu2_state state;
    return state;
}

static rknn_core_mask ggml_rknpu2_core_mask(int core) {
    switch (core) {
        case 0:  return RKNN_NPU_CORE_0;
        case 1:  return RKNN_NPU_CORE_1;
        case 2:  return RKNN_NPU_CORE_2;
        default: return RKNN_NPU_CORE_AUTO;
    }
}

// split the output columns over the NPU cores. Each core gets its own context and its
// own slice of B, so total weight traffic is unchanged and no reduction is needed.
static void ggml_rknpu2_plan_slices(const ggml_rknpu2_soc & soc, int64_t N, int n_cores,
                                    std::vector<ggml_rknpu2_slice> & slices) {
    slices.clear();

    const int64_t base = N / n_cores / soc.n_align * soc.n_align;
    if (n_cores <= 1 || base < soc.n_align) {
        slices.push_back({ 0, N, -1 });
        return;
    }

    int64_t n0 = 0;
    for (int i = 0; i < n_cores; i++) {
        const int64_t nc = (i == n_cores - 1) ? N - n0 : base;
        slices.push_back({ n0, nc, i });
        n0 += nc;
    }
}

// split K so every chunk is aligned and within the chip limit
static bool ggml_rknpu2_plan_chunks(const ggml_rknpu2_soc & soc, int64_t K, std::vector<int64_t> & chunks) {
    chunks.clear();

    if (K <= 0 || K % soc.k_align != 0) {
        return false;
    }
    if (K <= soc.k_max) {
        chunks.push_back(K);
        return true;
    }

    const int64_t n = (K + soc.k_max - 1) / soc.k_max;
    int64_t size = ((K + n - 1) / n + soc.k_align - 1) / soc.k_align * soc.k_align;
    if (size > soc.k_max || size <= 0) {
        return false;
    }

    int64_t left = K;
    while (left > 0) {
        const int64_t c = left < size ? left : size;
        if (c % soc.k_align != 0) {
            return false;
        }
        chunks.push_back(c);
        left -= c;
    }
    return true;
}

// cheap content sample, guards against a freed weight pointer being reused by a new tensor
static uint64_t ggml_rknpu2_fingerprint(const ggml_tensor * t) {
    const uint8_t * p = (const uint8_t *) t->data;
    const size_t    n = ggml_nbytes(t);

    uint64_t h = 1469598103934665603ull ^ (uint64_t) n;
    for (int i = 0; i < 8; i++) {
        const size_t off = (size_t) ((double) i / 8.0 * (double) (n > 8 ? n - 8 : 0));
        uint64_t     v   = 0;
        memcpy(&v, p + off, n >= off + 8 ? 8 : (n - off));
        h = (h ^ v) * 1099511628211ull;
    }
    return h;
}

// run one small matmul pinned to one core and check the result. True only if the runtime
// accepts the mask and the core computes correctly.
static bool ggml_rknpu2_core_works(const ggml_rknpu2_soc & soc, rknn_core_mask mask) {
    const int K = (int) soc.k_align;
    const int N = (int) soc.n_align;

    rknn_matmul_ctx     ctx;
    rknn_matmul_info    info;
    rknn_matmul_io_attr io_attr;

    memset(&info, 0, sizeof(info));
    info.M         = 1;
    info.K         = K;
    info.N         = N;
    info.type      = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
    info.B_layout  = RKNN_MM_LAYOUT_NATIVE;
    info.AC_layout = RKNN_MM_LAYOUT_NORM;

    if (rknn_matmul_create(&ctx, &info, &io_attr) != RKNN_SUCC) {
        return false;
    }

    bool ok = rknn_matmul_set_core_mask(ctx, mask) == RKNN_SUCC &&
              io_attr.A.size == (uint32_t) (K * 2) && io_attr.B.size == (uint32_t) (K * N * 2) && io_attr.C.size == (uint32_t) (N * 4);

    rknn_tensor_mem * A = ok ? rknn_create_mem(ctx, io_attr.A.size) : nullptr;
    rknn_tensor_mem * B = ok ? rknn_create_mem(ctx, io_attr.B.size) : nullptr;
    rknn_tensor_mem * C = ok ? rknn_create_mem(ctx, io_attr.C.size) : nullptr;
    ok = ok && A && B && C;

    if (ok) {
        // A is all 1.0, column n of B is 0.25*(n%8+1), so C[n] = K*0.25*(n%8+1), exact in fp16/fp32
        std::vector<ggml_fp16_t> norm((size_t) K * N);
        for (int k = 0; k < K; k++) {
            for (int n = 0; n < N; n++) {
                norm[(size_t) k * N + n] = ggml_fp32_to_fp16(0.25f * (n % 8 + 1));
            }
        }
        ggml_fp16_t * a = (ggml_fp16_t *) A->virt_addr;
        for (int k = 0; k < K; k++) {
            a[k] = ggml_fp32_to_fp16(1.0f);
        }
        memset(C->virt_addr, 0, io_attr.C.size);

        ok = rknn_B_normal_layout_to_native_layout(norm.data(), B->virt_addr, K, N, &info) == RKNN_SUCC &&
             rknn_matmul_set_io_mem(ctx, A, &io_attr.A) == RKNN_SUCC &&
             rknn_matmul_set_io_mem(ctx, B, &io_attr.B) == RKNN_SUCC &&
             rknn_matmul_set_io_mem(ctx, C, &io_attr.C) == RKNN_SUCC &&
             rknn_mem_sync(ctx, A, RKNN_MEMORY_SYNC_TO_DEVICE) == RKNN_SUCC &&
             rknn_mem_sync(ctx, B, RKNN_MEMORY_SYNC_TO_DEVICE) == RKNN_SUCC &&
             rknn_matmul_run(ctx) == RKNN_SUCC &&
             rknn_mem_sync(ctx, C, RKNN_MEMORY_SYNC_FROM_DEVICE) == RKNN_SUCC;

        const float * c = (const float *) C->virt_addr;
        for (int n = 0; ok && n < N; n++) {
            ok = c[n] == K * 0.25f * (n % 8 + 1);
        }
    }

    if (A) rknn_destroy_mem(ctx, A);
    if (B) rknn_destroy_mem(ctx, B);
    if (C) rknn_destroy_mem(ctx, C);
    rknn_matmul_destroy(ctx);
    return ok;
}

static void ggml_rknpu2_probe(ggml_rknpu2_state & state) {
    if (state.availability != -1) {
        return;
    }

    state.soc         = ggml_rknpu2_detect_soc();
    state.m_min       = ggml_rknpu2_env_i64("GGML_RKNPU2_MIN_M", 128);
    state.m_max       = ggml_rknpu2_env_i64("GGML_RKNPU2_MAX_M", 512);
    // every context uses iommu_domain_id 0, which the API header limits to 4GB. Stop below
    // that, so a model too big for the NPU leaves its remaining tensors on the CPU backend
    // instead of allocating until rknn_create_mem fails. More domains would raise the cap.
    state.bytes_cap   = (size_t) ggml_rknpu2_env_i64("GGML_RKNPU2_MEM_MB", 3072) * 1024 * 1024;
    state.f32_weights = getenv("GGML_RKNPU2_F32_WEIGHTS") != nullptr;
    state.n_cores     = (int) ggml_rknpu2_env_i64("GGML_RKNPU2_CORES", state.soc.n_cores);

    if (state.n_cores < 1) state.n_cores = 1;
    if (state.n_cores > state.soc.n_cores) state.n_cores = state.soc.n_cores;

    if (state.m_max < state.m_min) {
        state.m_max = state.m_min;
    }

    rknn_matmul_ctx     ctx;
    rknn_matmul_info    info;
    rknn_matmul_io_attr io_attr;

    memset(&info, 0, sizeof(info));
    info.M         = 1;
    info.K         = (int32_t) state.soc.k_align;
    info.N         = (int32_t) state.soc.n_align;
    info.type      = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
    info.B_layout  = RKNN_MM_LAYOUT_NATIVE;
    info.AC_layout = RKNN_MM_LAYOUT_NORM;

    if (rknn_matmul_create(&ctx, &info, &io_attr) != RKNN_SUCC) {
        GGML_LOG_INFO("%s: no usable Rockchip NPU found, backend disabled\n", __func__);
        state.availability = 0;
        return;
    }

    rknn_sdk_version ver;
    memset(&ver, 0, sizeof(ver));
    if (rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &ver, sizeof(ver)) == RKNN_SUCC) {
        GGML_LOG_INFO("%s: rknn api %s, driver %s\n", __func__, ver.api_version, ver.drv_version);
    }
    rknn_matmul_destroy(ctx);

    // pin each slice to its own core if the runtime really supports it on this chip.
    // Otherwise the slices still run in parallel and the driver puts each job on a free core.
    state.pin_cores = false;
    if (state.n_cores > 1) {
        state.pin_cores = true;
        for (int i = 0; i < state.n_cores && state.pin_cores; i++) {
            state.pin_cores = ggml_rknpu2_core_works(state.soc, ggml_rknpu2_core_mask(i));
        }
        if (!state.pin_cores) {
            GGML_LOG_INFO("%s: runtime does not accept a core mask on %s, running %d contexts in parallel without pinning\n",
                          __func__, state.soc.name, state.n_cores);
        }
    }

    state.workers.start((size_t) state.n_cores - 1);

    GGML_LOG_INFO("%s: SoC %s, K align %d, N align %d, K max %d, cores %d (%s)\n", __func__,
                  state.soc.name, (int) state.soc.k_align, (int) state.soc.n_align,
                  (int) state.soc.k_max, state.n_cores,
                  state.n_cores == 1 ? "single" : state.pin_cores ? "pinned" : "parallel");

    state.availability = 1;
}

// caller holds the mutex
static ggml_rknpu2_kernel * ggml_rknpu2_get_kernel(ggml_rknpu2_state & state, int64_t Kc, int64_t N, int core) {
    const ggml_rknpu2_shape shape(Kc, N, core);

    auto it = state.kernels.find(shape);
    if (it != state.kernels.end()) {
        return &it->second;
    }
    if (state.rejected.count(shape)) {
        return nullptr;
    }

    // a weight holds B buffers created against this context, so the tile is fixed for the
    // model lifetime. It must not come from the first batch seen, or a small warmup batch
    // pins a small tile for the whole run. rknn_matmul_create_dynamic_shape would allow
    // several M per context and remove the padding below the tile.
    const int64_t m_tile = state.m_max;

    ggml_rknpu2_kernel k;
    k.m_tile = m_tile;

    memset(&k.info, 0, sizeof(k.info));
    k.info.M         = (int32_t) m_tile;
    k.info.K         = (int32_t) Kc;
    k.info.N         = (int32_t) N;
    k.info.type      = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
    k.info.B_layout  = RKNN_MM_LAYOUT_NATIVE;
    k.info.AC_layout = RKNN_MM_LAYOUT_NORM;

    if (rknn_matmul_create(&k.ctx, &k.info, &k.io_attr) != RKNN_SUCC) {
        GGML_LOG_DEBUG("%s: runtime refused M=%d K=%d N=%d, keeping it on CPU\n", __func__,
                       (int) m_tile, (int) Kc, (int) N);
        state.rejected.insert(shape);
        return nullptr;
    }

    // we index A and C as plain row major and convert B as one flat native buffer, refuse the shape if the runtime disagrees
    if (k.io_attr.A.size != (uint32_t) (m_tile * Kc * 2) || k.io_attr.C.size != (uint32_t) (m_tile * N * 4) ||
        k.io_attr.B.size != (uint32_t) (Kc * N * 2)) {
        GGML_LOG_WARN("%s: unexpected A/B/C layout for M=%d K=%d N=%d, keeping it on CPU\n", __func__,
                      (int) m_tile, (int) Kc, (int) N);
        rknn_matmul_destroy(k.ctx);
        state.rejected.insert(shape);
        return nullptr;
    }

    if (state.pin_cores && core >= 0 && rknn_matmul_set_core_mask(k.ctx, ggml_rknpu2_core_mask(core)) != RKNN_SUCC) {
        GGML_LOG_WARN("%s: core mask %d refused for M=%d K=%d N=%d, keeping it on CPU\n", __func__,
                      core, (int) m_tile, (int) Kc, (int) N);
        rknn_matmul_destroy(k.ctx);
        state.rejected.insert(shape);
        return nullptr;
    }

    k.A = rknn_create_mem(k.ctx, k.io_attr.A.size);
    k.C = rknn_create_mem(k.ctx, k.io_attr.C.size);

    if (k.A == nullptr || k.C == nullptr ||
        rknn_matmul_set_io_mem(k.ctx, k.A, &k.io_attr.A) != RKNN_SUCC ||
        rknn_matmul_set_io_mem(k.ctx, k.C, &k.io_attr.C) != RKNN_SUCC) {
        GGML_LOG_WARN("%s: failed to set up NPU memory for M=%d K=%d N=%d\n", __func__,
                      (int) m_tile, (int) Kc, (int) N);
        if (k.A) rknn_destroy_mem(k.ctx, k.A);
        if (k.C) rknn_destroy_mem(k.ctx, k.C);
        rknn_matmul_destroy(k.ctx);
        state.rejected.insert(shape);
        return nullptr;
    }

    memset(k.A->virt_addr, 0, k.io_attr.A.size);
    state.bytes_used += k.io_attr.A.size + k.io_attr.C.size;

    auto res = state.kernels.emplace(shape, k);
    return &res.first->second;
}

// dequantize one weight row to float, works for F32, F16 and every quantized type
static void ggml_rknpu2_row_to_float(const ggml_tensor * src0, int64_t row, float * dst, int64_t K) {
    const char * p = (const char *) src0->data + row * src0->nb[1];

    if (src0->type == GGML_TYPE_F32) {
        memcpy(dst, p, K * sizeof(float));
        return;
    }
    if (src0->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) p, dst, K);
        return;
    }
    ggml_get_type_traits(src0->type)->to_float(p, dst, K);
}

static void ggml_rknpu2_free_weight(ggml_rknpu2_weight & w) {
    for (size_t s = 0; s < w.B.size(); s++) {
        for (size_t i = 0; i < w.B[s].size(); i++) {
            rknn_destroy_mem(w.ctxs[s][i], w.B[s][i]);
        }
    }
    w.B.clear();
    w.ctxs.clear();
}

// NPU bytes one weight takes, summed over its column slices and K chunks
static size_t ggml_rknpu2_weight_bytes(const std::vector<std::vector<ggml_rknpu2_kernel *>> & kernels) {
    size_t n = 0;
    for (const auto & ks : kernels) {
        for (const ggml_rknpu2_kernel * k : ks) {
            n += k->io_attr.B.size;
        }
    }
    return n;
}

// caller holds the mutex
static ggml_rknpu2_weight * ggml_rknpu2_get_weight(ggml_rknpu2_state & state, const ggml_tensor * src0,
                                                   const std::vector<int64_t> & chunks,
                                                   const std::vector<ggml_rknpu2_slice> & slices,
                                                   std::vector<std::vector<ggml_rknpu2_kernel *>> & kernels) {
    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];

    const ggml_rknpu2_weight_key key = { src0->data, K, N, src0->type };
    const uint64_t               fp  = ggml_rknpu2_fingerprint(src0);

    auto it = state.weights.find(key);
    if (it != state.weights.end()) {
        if (it->second.fingerprint == fp) {
            return &it->second;
        }
        // the address was reused by different data, drop the stale copy
        ggml_rknpu2_free_weight(it->second);
        state.bytes_used -= it->second.bytes;
        state.weights.erase(it);
        // a new B buffer can get the virtual address of a freed one, so force a rebind
        for (auto & kit : state.kernels) {
            kit.second.bound_B = nullptr;
        }
    }

    // the reservation made by supports_op becomes real usage below, or is dropped on failure
    size_t reserved = 0;
    auto   rit      = state.reserved.find(key);
    if (rit != state.reserved.end()) {
        reserved = rit->second;
        state.bytes_reserved -= reserved;
        state.reserved.erase(rit);
    }

    const size_t need = ggml_rknpu2_weight_bytes(kernels);
    if (state.bytes_cap > 0 && reserved < need && state.bytes_used + state.bytes_reserved + need > state.bytes_cap) {
        GGML_LOG_WARN("%s: a %lld x %lld weight does not fit in GGML_RKNPU2_MEM_MB\n", __func__, (long long) K, (long long) N);
        return nullptr;
    }

    ggml_rknpu2_weight w;
    w.fingerprint = fp;
    w.B.resize(slices.size());
    w.ctxs.resize(slices.size());

    std::vector<float>       row(K);
    std::vector<ggml_fp16_t> norm;

    for (size_t s = 0; s < slices.size(); s++) {
        const int64_t n0 = slices[s].n0;
        const int64_t nc = slices[s].nc;

        // B normal layout is (K, nc), src0 rows are the output channels of this slice. Each
        // K chunk is a row range of it, so the slice is dequantized once for all chunks.
        norm.resize((size_t) K * nc);
        for (int64_t j = 0; j < nc; j++) {
            ggml_rknpu2_row_to_float(src0, n0 + j, row.data(), K);
            for (int64_t k = 0; k < K; k++) {
                norm[(size_t) k * nc + j] = ggml_fp32_to_fp16(row[k]);
            }
        }

        int64_t k0 = 0;
        for (size_t c = 0; c < chunks.size(); c++) {
            const int64_t Kc = chunks[c];
            ggml_rknpu2_kernel * kernel = kernels[s][c];

            rknn_tensor_mem * B = rknn_create_mem(kernel->ctx, kernel->io_attr.B.size);
            if (B == nullptr) {
                GGML_LOG_WARN("%s: out of NPU memory for a %lld x %lld weight\n", __func__, (long long) K, (long long) N);
                ggml_rknpu2_free_weight(w);
                return nullptr;
            }

            if (rknn_B_normal_layout_to_native_layout(norm.data() + (size_t) k0 * nc, B->virt_addr, (int) Kc, (int) nc, &kernel->info) != RKNN_SUCC ||
                rknn_mem_sync(kernel->ctx, B, RKNN_MEMORY_SYNC_TO_DEVICE) != RKNN_SUCC) {
                GGML_LOG_WARN("%s: weight conversion failed for K=%lld N=%lld\n", __func__, (long long) Kc, (long long) nc);
                rknn_destroy_mem(kernel->ctx, B);
                ggml_rknpu2_free_weight(w);
                return nullptr;
            }

            w.B[s].push_back(B);
            w.ctxs[s].push_back(kernel->ctx);
            w.bytes += kernel->io_attr.B.size;
            k0 += Kc;
        }
    }

    state.bytes_used += w.bytes;

    auto res = state.weights.emplace(key, w);
    return &res.first->second;
}

// correctness net for the rare case the NPU path gives up after the scheduler committed
static void ggml_rknpu2_mul_mat_cpu(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, int64_t M) {
    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];

    float * d = (float *) dst->data;

    // ponytail: plain loops, this only runs when the NPU refuses work it already accepted
    std::vector<float> row(K);
    for (int64_t n = 0; n < N; n++) {
        ggml_rknpu2_row_to_float(src0, n, row.data(), K);
        for (int64_t m = 0; m < M; m++) {
            const float * x = (const float *) src1->data + m * K;
            float acc = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                acc += row[k] * x[k];
            }
            d[m * N + n] = acc;
        }
    }
}

// run every tile and K chunk for one column slice, on one NPU core
static bool ggml_rknpu2_run_slice(const ggml_rknpu2_slice & slice, const std::vector<int64_t> & chunks,
                                  std::vector<ggml_rknpu2_kernel *> & kernels,
                                  const std::vector<rknn_tensor_mem *> & B,
                                  const ggml_tensor * src1, ggml_tensor * dst,
                                  int64_t M, int64_t K, int64_t N, int64_t m_step) {
    std::vector<float> row_scale((size_t) m_step, 1.0f);

    for (int64_t m0 = 0; m0 < M; m0 += m_step) {
        const int64_t rows = (M - m0) < m_step ? (M - m0) : m_step;

        int64_t k0 = 0;
        for (size_t c = 0; c < chunks.size(); c++) {
            const int64_t        Kc     = chunks[c];
            ggml_rknpu2_kernel * kernel = kernels[c];

            ggml_fp16_t * A = (ggml_fp16_t *) kernel->A->virt_addr;
            for (int64_t m = 0; m < rows; m++) {
                const float * x = (const float *) src1->data + (m0 + m) * K + k0;

                float amax = 0.0f;
                for (int64_t k = 0; k < Kc; k++) {
                    amax = fmaxf(amax, fabsf(x[k]));
                }

                // fp16 overflows above 65504, scale such rows by a power of two and undo it on C
                if (amax < 65504.0f || !std::isfinite(amax)) {
                    ggml_fp32_to_fp16_row(x, A + m * Kc, Kc);
                    row_scale[m] = 1.0f;
                } else {
                    int e;
                    frexpf(amax / 32768.0f, &e);
                    const float s = ldexpf(1.0f, -e);
                    for (int64_t k = 0; k < Kc; k++) {
                        A[m * Kc + k] = ggml_fp32_to_fp16(x[k] * s);
                    }
                    row_scale[m] = ldexpf(1.0f, e);
                }
            }
            if (rows < kernel->m_tile) {
                memset(A + rows * Kc, 0, (size_t) (kernel->m_tile - rows) * Kc * sizeof(ggml_fp16_t));
            }

            if (kernel->bound_B != B[c]->virt_addr) {
                if (rknn_matmul_set_io_mem(kernel->ctx, B[c], &kernel->io_attr.B) != RKNN_SUCC) {
                    return false;
                }
                kernel->bound_B = B[c]->virt_addr;
            }

            // NPU memory is cacheable, the CPU writes to A must reach the device
            if (rknn_mem_sync(kernel->ctx, kernel->A, RKNN_MEMORY_SYNC_TO_DEVICE) != RKNN_SUCC ||
                rknn_matmul_run(kernel->ctx) != RKNN_SUCC ||
                rknn_mem_sync(kernel->ctx, kernel->C, RKNN_MEMORY_SYNC_FROM_DEVICE) != RKNN_SUCC) {
                return false;
            }

            // this slice owns columns [n0, n0+nc) of every row, other slices own the rest
            const float * C = (const float *) kernel->C->virt_addr;
            for (int64_t m = 0; m < rows; m++) {
                float *       d  = (float *) dst->data + (m0 + m) * N + slice.n0;
                const float * cm = C + m * slice.nc;
                const float   rs = row_scale[m];
                if (c == 0 && rs == 1.0f) {
                    memcpy(d, cm, (size_t) slice.nc * sizeof(float));
                } else if (c == 0) {
                    for (int64_t i = 0; i < slice.nc; i++) {
                        d[i] = cm[i] * rs;
                    }
                } else {
                    for (int64_t i = 0; i < slice.nc; i++) {
                        d[i] += cm[i] * rs;
                    }
                }
            }
            k0 += Kc;
        }
    }
    return true;
}

static void ggml_rknpu2_mul_mat(ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    // src0 is 2D so every src1 column is an independent row of one flat matmul
    const int64_t M = src1->ne[1] * src1->ne[2] * src1->ne[3];

    ggml_rknpu2_state & state = ggml_rknpu2_get_state();
    std::lock_guard<std::mutex> lock(state.mutex);

    std::vector<int64_t>           chunks;
    std::vector<ggml_rknpu2_slice> slices;
    std::vector<std::vector<ggml_rknpu2_kernel *>> kernels;

    ggml_rknpu2_plan_slices(state.soc, N, state.n_cores, slices);

    bool ok = ggml_rknpu2_plan_chunks(state.soc, K, chunks);
    if (ok) {
        kernels.resize(slices.size());
        for (size_t s = 0; s < slices.size() && ok; s++) {
            for (int64_t Kc : chunks) {
                ggml_rknpu2_kernel * k = ggml_rknpu2_get_kernel(state, Kc, slices[s].nc, slices[s].core);
                if (k == nullptr) {
                    ok = false;
                    break;
                }
                kernels[s].push_back(k);
            }
        }
    }

    ggml_rknpu2_weight * w = ok ? ggml_rknpu2_get_weight(state, src0, chunks, slices, kernels) : nullptr;
    if (w == nullptr) {
        // the scheduler already gave us this node, so this call must take the scalar path.
        // Remember the weight, so supports_op declines it and the CPU backend gets it next
        // time. Without this every later call retries the same failed allocation.
        const ggml_rknpu2_weight_key key = { src0->data, K, N, src0->type };
        state.rejected_weights.insert(key);
        auto rit = state.reserved.find(key);
        if (rit != state.reserved.end()) {
            state.bytes_reserved -= rit->second;
            state.reserved.erase(rit);
        }
        if (!state.warned_fallback) {
            GGML_LOG_WARN("%s: could not place a %lld x %lld weight on the NPU, this one matmul "
                          "runs on the slow CPU path and the tensor stays on the CPU backend "
                          "after that (raise GGML_RKNPU2_MEM_MB if this is unexpected)\n",
                          __func__, (long long) K, (long long) N);
            state.warned_fallback = true;
        }
        ggml_rknpu2_mul_mat_cpu(src0, src1, dst, M);
        return;
    }

    // chunks can land on kernels built at different times, so step by the smallest tile
    int64_t m_step = kernels[0][0]->m_tile;
    for (const auto & ks : kernels) {
        for (const ggml_rknpu2_kernel * k : ks) {
            if (k->m_tile < m_step) {
                m_step = k->m_tile;
            }
        }
    }

    // slice 0 runs on the calling thread, the rest on the pooled workers
    std::vector<char> done(slices.size(), 0);

    state.workers.run(slices.size() - 1, [&](size_t s) {
        done[s] = ggml_rknpu2_run_slice(slices[s], chunks, kernels[s], w->B[s], src1, dst, M, K, N, m_step) ? 1 : 0;
    });

    for (char d : done) {
        if (!d) {
            GGML_LOG_WARN("%s: NPU matmul failed, using the slow CPU path\n", __func__);
            ggml_rknpu2_mul_mat_cpu(src0, src1, dst, M);
            return;
        }
    }
}

// backend interface

static const char * ggml_backend_rknpu2_get_name(ggml_backend_t backend) {
    return "RKNPU2";

    GGML_UNUSED(backend);
}

static void ggml_backend_rknpu2_free(ggml_backend_t backend) {
    ggml_rknpu2_state & state = ggml_rknpu2_get_state();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (--state.n_backends == 0) {
            for (auto & it : state.weights) {
                ggml_rknpu2_free_weight(it.second);
            }
            state.weights.clear();
            for (auto & it : state.kernels) {
                rknn_destroy_mem(it.second.ctx, it.second.A);
                rknn_destroy_mem(it.second.ctx, it.second.C);
                rknn_matmul_destroy(it.second.ctx);
            }
            state.kernels.clear();
            state.rejected.clear();
            state.rejected_weights.clear();
            state.reserved.clear();
            state.workers.shutdown();
            state.availability   = -1;
            state.bytes_used     = 0;
            state.bytes_reserved = 0;
        }
    }
    delete backend;
}

static enum ggml_status ggml_backend_rknpu2_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ggml_rknpu2_mul_mat(node);
                break;

            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;

            default:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }

    return GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

static struct ggml_backend_i rknpu2_backend_i = {
    /* .get_name                = */ ggml_backend_rknpu2_get_name,
    /* .free                    = */ ggml_backend_rknpu2_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_rknpu2_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_rknpu2_guid(void) {
    static ggml_guid guid = { 0x7b, 0x14, 0x9a, 0x3e, 0xd2, 0x50, 0x4c, 0x81, 0xb6, 0x0f, 0x27, 0xe8, 0x93, 0x5a, 0xc4, 0x66 };
    return &guid;
}

ggml_backend_t ggml_backend_rknpu2_init(void) {
    ggml_rknpu2_state & state = ggml_rknpu2_get_state();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        ggml_rknpu2_probe(state);
        state.n_backends++;
    }

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_rknpu2_guid(),
        /* .iface   = */ rknpu2_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_rknpu2_reg(), 0),
        /* .context = */ nullptr,
    };

    return backend;
}

bool ggml_backend_is_rknpu2(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_rknpu2_guid());
}

// device interface

static const char * ggml_backend_rknpu2_device_get_name(ggml_backend_dev_t dev) {
    return "RKNPU2";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_rknpu2_device_get_description(ggml_backend_dev_t dev) {
    static std::string desc = [] {
        ggml_rknpu2_state & state = ggml_rknpu2_get_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        ggml_rknpu2_probe(state);
        return std::string("Rockchip NPU (") + state.soc.name + ")";
    }();
    return desc.c_str();

    GGML_UNUSED(dev);
}

static void ggml_backend_rknpu2_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // the NPU shares system memory, nothing separate to report
    *free  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_rknpu2_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

static void ggml_backend_rknpu2_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rknpu2_device_get_name(dev);
    props->description = ggml_backend_rknpu2_device_get_description(dev);
    props->type        = ggml_backend_rknpu2_device_get_type(dev);
    ggml_backend_rknpu2_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
        /* .mmap_support          = */ true,
    };
}

static ggml_backend_t ggml_backend_rknpu2_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_rknpu2_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_rknpu2_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_rknpu2_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_rknpu2_supports_mul_mat(const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    if (src1->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        return false;
    }
    // src0 must be a plain 2D weight, src1 and dst flat, so the whole thing is one matmul
    if (src0->ne[2] != 1 || src0->ne[3] != 1) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(op)) {
        return false;
    }
    if (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
        return false;
    }
    // the weight copy on the NPU is cached, so the data must not change between runs
    if (src0->buffer == NULL || src0->buffer->usage != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        return false;
    }
    // no_alloc model loads (llama.cpp memory fit) have weights without data, nothing to cache
    if (src0->data == NULL) {
        return false;
    }

    ggml_rknpu2_state & state = ggml_rknpu2_get_state();
    std::lock_guard<std::mutex> lock(state.mutex);

    ggml_rknpu2_probe(state);
    if (state.availability != 1) {
        return false;
    }

    if (src0->type == GGML_TYPE_F32) {
        // the NPU only does fp16, that would silently lose precision the CPU keeps
        if (!state.f32_weights) {
            if (!state.warned_f32) {
                GGML_LOG_INFO("%s: skipping F32 weights, set GGML_RKNPU2_F32_WEIGHTS=1 to offload them in fp16\n", __func__);
                state.warned_f32 = true;
            }
            return false;
        }
    } else if (src0->type != GGML_TYPE_F16 && ggml_get_type_traits(src0->type)->to_float == NULL) {
        return false;
    }

    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = src1->ne[1] * src1->ne[2] * src1->ne[3];

    if (M < state.m_min || N % state.soc.n_align != 0) {
        return false;
    }

    std::vector<int64_t> chunks;
    if (!ggml_rknpu2_plan_chunks(state.soc, K, chunks)) {
        return false;
    }
    std::vector<ggml_rknpu2_slice> slices;
    ggml_rknpu2_plan_slices(state.soc, N, state.n_cores, slices);

    const ggml_rknpu2_weight_key key = { src0->data, K, N, src0->type };

    // a weight that already failed to land on the NPU is never offered again
    if (state.rejected_weights.count(key)) {
        return false;
    }

    // build the contexts here rather than at compute time. The documented shape limits are
    // not exact for every chip, and once the scheduler hands us a node we can only fall
    // back to a slow scalar loop. Failing here instead sends the node to the CPU backend
    // and its fast kernels. Contexts are cached, so this runs once per shape.
    std::vector<std::vector<ggml_rknpu2_kernel *>> kernels(slices.size());
    for (size_t s = 0; s < slices.size(); s++) {
        for (int64_t Kc : chunks) {
            ggml_rknpu2_kernel * k = ggml_rknpu2_get_kernel(state, Kc, slices[s].nc, slices[s].core);
            if (k == nullptr) {
                return false;
            }
            kernels[s].push_back(k);
        }
    }

    // the scheduler asks for every node before any of them runs, so room is reserved here.
    // Checking only the bytes already uploaded would accept every weight of a big model.
    if (state.bytes_cap > 0 && state.weights.find(key) == state.weights.end() && state.reserved.find(key) == state.reserved.end()) {
        const size_t need = ggml_rknpu2_weight_bytes(kernels);
        if (state.bytes_used + state.bytes_reserved + need > state.bytes_cap) {
            return false;
        }
        state.reserved[key]   = need;
        state.bytes_reserved += need;
    }

    return true;
}

static bool ggml_backend_rknpu2_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_MUL_MAT:
            return ggml_backend_rknpu2_supports_mul_mat(op);

        default:
            return false;
    }

    GGML_UNUSED(dev);
}

static bool ggml_backend_rknpu2_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (ggml_backend_buft_is_host(buft)) {
        return true;
    }

    // repacked weights use a CPU only layout, so the NPU never gets them
    static std::atomic<bool> warned_repack{false};
    if (strcmp(ggml_backend_buft_name(buft), "CPU_REPACK") == 0 && !warned_repack.exchange(true)) {
        GGML_LOG_INFO("%s: weights in CPU_REPACK buffers stay on the CPU, run with --no-repack (llama-bench: --repack 0) to offload them to the NPU\n", __func__);
    }
    return false;

    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_rknpu2_device_i = {
    /* .get_name             = */ ggml_backend_rknpu2_device_get_name,
    /* .get_description      = */ ggml_backend_rknpu2_device_get_description,
    /* .get_memory           = */ ggml_backend_rknpu2_device_get_memory,
    /* .get_type             = */ ggml_backend_rknpu2_device_get_type,
    /* .get_props            = */ ggml_backend_rknpu2_device_get_props,
    /* .init_backend         = */ ggml_backend_rknpu2_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_rknpu2_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_rknpu2_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_rknpu2_device_supports_op,
    /* .supports_buft        = */ ggml_backend_rknpu2_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// backend reg interface

static const char * ggml_backend_rknpu2_reg_get_name(ggml_backend_reg_t reg) {
    return "RKNPU2";

    GGML_UNUSED(reg);
}

// only list the NPU when it can be used, so --list-devices tells the truth on a board
// without the driver or without permission (a backend built in, not loaded at run time,
// is registered either way)
static size_t ggml_backend_rknpu2_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_rknpu2_state & state = ggml_rknpu2_get_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    ggml_rknpu2_probe(state);
    return state.availability == 1 ? 1 : 0;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_rknpu2_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_device ggml_backend_rknpu2_device = {
        /* .iface   = */ ggml_backend_rknpu2_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    return &ggml_backend_rknpu2_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

static const struct ggml_backend_reg_i ggml_backend_rknpu2_reg_i = {
    /* .get_name         = */ ggml_backend_rknpu2_reg_get_name,
    /* .get_device_count = */ ggml_backend_rknpu2_reg_get_device_count,
    /* .get_device       = */ ggml_backend_rknpu2_reg_get_device,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_rknpu2_reg(void) {
    static struct ggml_backend_reg ggml_backend_rknpu2_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rknpu2_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_rknpu2_reg;
}

static int ggml_backend_rknpu2_score(void) {
    ggml_rknpu2_state & state = ggml_rknpu2_get_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    ggml_rknpu2_probe(state);
    return state.availability == 1 ? 1 : 0;
}

GGML_BACKEND_DL_IMPL(ggml_backend_rknpu2_reg)
GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_rknpu2_score)
