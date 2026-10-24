// CPU stub of the librknnrt matmul API, used to test ggml-rknpu2 off-device.
//
// Deliberately strict, it fails the same way real hardware would:
//  - memory is NON COHERENT. The device only sees what rknn_mem_sync(TO_DEVICE)
//    pushed, and the CPU only sees what rknn_mem_sync(FROM_DEVICE) pulled back.
//    A missing sync gives wrong results instead of passing by luck.
//  - B really is stored in the (N/na, K/ka, na, ka) native layout and decoded
//    again at run time, so a K/N mix up between create and convert shows up.
//  - shapes that violate the selected chip limits are rejected by create.
//
// Chip is selected with GGML_RKNPU2_SOC, same variable the backend uses.

#include "rknn_api.h"
#include "rknn_matmul_api.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <chrono>
#include <map>
#include <thread>
#include <mutex>
#include <string>
#include <vector>

struct stub_mem {
    std::vector<uint8_t> host;   // what the CPU sees (virt_addr)
    std::vector<uint8_t> device; // what the NPU sees
};

struct stub_ctx {
    rknn_matmul_info info;
    rknn_core_mask   core = RKNN_NPU_CORE_AUTO;
    rknn_tensor_mem * A = nullptr;
    rknn_tensor_mem * B = nullptr;
    rknn_tensor_mem * C = nullptr;
};

static std::map<uint64_t, stub_ctx>            g_ctxs;
static std::map<rknn_tensor_mem *, stub_mem *> g_mems;
static uint64_t                                g_next = 1;

// the backend drives one context per NPU core in parallel, so the tables are shared
static std::mutex g_stub_mutex;

std::atomic<int> g_stub_runs{0};      // test visible counters
std::atomic<int> g_stub_work_runs{0}; // runs with M > 1, i.e. not the backend's startup self test
std::atomic<int> g_stub_core_mask{0}; // OR of every core mask that actually ran

// ksplit: above this K the real B native layout is split into T segments (rknn_matmul_api.h),
// which this stub does not model. 0 means no documented split.
// cores: NPU cores of the chip, a mask naming any other core is refused.
struct stub_soc { const char * name; int ka; int na; int kmax; int ksplit; int cores; };

static const stub_soc g_socs[] = {
    { "rk3588", 32, 16, 10240, 8192, 3 },
    { "rk3576", 32, 16, 10240, 4096, 2 },
    { "rk3566", 16,  8, 10240, 0,    1 },
    { "rk3568", 16,  8, 10240, 0,    1 },
    { "rk3562", 32,  8, 10240, 0,    1 },
};

std::atomic<int> g_stub_inflight{0};
std::atomic<int> g_stub_max_inflight{0}; // most runs active at once, shows real parallel dispatch

// STUB_REPORT=1 prints the run count at exit, so an end-to-end test can tell a model
// really went through the NPU path and not only through the CPU backend
static struct stub_report {
    ~stub_report() {
        if (getenv("STUB_REPORT")) {
            fprintf(stderr, "stub: %d matmul runs, %d with M > 1\n", g_stub_runs.load(), g_stub_work_runs.load());
        }
    }
} g_stub_report;

static stub_soc stub_get_soc(void) {
    const char * env = getenv("GGML_RKNPU2_SOC");
    std::string  id  = env ? env : "rk3588";
    for (const stub_soc & s : g_socs) {
        if (id.find(s.name) != std::string::npos) {
            return s;
        }
    }
    return g_socs[0];
}

static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t man  = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((man & 0x400) == 0) { man <<= 1; exp--; }
            man &= 0x3ff;
            bits = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000 | (man << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

// index of element (k, n) inside the (N/na, K/ka, na, ka) native B buffer
static size_t native_index(int k, int n, int K, int ka, int na) {
    const int kb = k / ka, ko = k % ka;
    const int nb = n / na, no = n % na;
    return ((size_t)(nb * (K / ka) + kb) * na + no) * ka + ko;
}

extern "C" {

int rknn_matmul_create(rknn_matmul_ctx * ctx, rknn_matmul_info * info, rknn_matmul_io_attr * io_attr) {
    std::lock_guard<std::mutex> lock(g_stub_mutex);
    const stub_soc soc = stub_get_soc();

    // fault injection: a board without the NPU driver, or without permission to open it
    if (const char * no_npu = getenv("STUB_NO_NPU")) {
        if (atoi(no_npu) != 0) return RKNN_ERR_DEVICE_UNAVAILABLE;
    }

    if (info->type != RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32) return RKNN_ERR_PARAM_INVALID;
    if (info->B_layout != RKNN_MM_LAYOUT_NATIVE)          return RKNN_ERR_PARAM_INVALID;
    if (info->AC_layout != RKNN_MM_LAYOUT_NORM)           return RKNN_ERR_PARAM_INVALID;
    if (info->M <= 0 || info->K <= 0 || info->N <= 0)     return RKNN_ERR_PARAM_INVALID;
    if (info->K % soc.ka != 0 || info->N % soc.na != 0)   return RKNN_ERR_PARAM_INVALID;
    if (info->K > soc.kmax)                               return RKNN_ERR_PARAM_INVALID;

    // fault injection: pretend the runtime refuses a shape the backend thought was fine
    if (const char * fail_k = getenv("STUB_FAIL_K")) {
        if (info->K == atoi(fail_k)) return RKNN_ERR_FAIL;
    }

    stub_ctx sc;
    sc.info = *info;

    memset(io_attr, 0, sizeof(*io_attr));
    strcpy(io_attr->A.name, "A");
    io_attr->A.size = (uint32_t)(info->M * info->K * 2);
    io_attr->A.type = RKNN_TENSOR_FLOAT16;
    strcpy(io_attr->B.name, "B");
    io_attr->B.size = (uint32_t)(info->K * info->N * 2);
    io_attr->B.type = RKNN_TENSOR_FLOAT16;
    strcpy(io_attr->C.name, "C");
    io_attr->C.size = (uint32_t)(info->M * info->N * 4);
    io_attr->C.type = RKNN_TENSOR_FLOAT32;

    *ctx = g_next++;
    g_ctxs[*ctx] = sc;
    return RKNN_SUCC;
}

int rknn_matmul_set_io_mem(rknn_matmul_ctx ctx, rknn_tensor_mem * mem, rknn_matmul_tensor_attr * attr) {
    std::lock_guard<std::mutex> lock(g_stub_mutex);
    auto it = g_ctxs.find(ctx);
    if (it == g_ctxs.end()) return RKNN_ERR_CTX_INVALID;
    if (g_mems.find(mem) == g_mems.end()) return RKNN_ERR_PARAM_INVALID;

    if      (strcmp(attr->name, "A") == 0) it->second.A = mem;
    else if (strcmp(attr->name, "B") == 0) it->second.B = mem;
    else if (strcmp(attr->name, "C") == 0) it->second.C = mem;
    else return RKNN_ERR_PARAM_INVALID;
    return RKNN_SUCC;
}

int rknn_matmul_set_core_mask(rknn_matmul_ctx ctx, rknn_core_mask mask) {
    std::lock_guard<std::mutex> lock(g_stub_mutex);
    auto it = g_ctxs.find(ctx);
    if (it == g_ctxs.end()) return RKNN_ERR_CTX_INVALID;

    // fault injection: a runtime that does not support the matmul core mask on this chip
    const char * no_mask = getenv("STUB_NO_CORE_MASK");
    if (no_mask && atoi(no_mask) != 0) return RKNN_ERR_FAIL;

    const stub_soc soc = stub_get_soc();
    if (soc.cores < 2) return RKNN_ERR_FAIL;
    if ((int) mask & ~((1 << soc.cores) - 1)) return RKNN_ERR_PARAM_INVALID;
    it->second.core = mask;
    return RKNN_SUCC;
}

int rknn_matmul_run(rknn_matmul_ctx ctx) {
    const uint16_t * A;
    const uint16_t * B;
    float *          C;
    int              M, K, N;
    rknn_core_mask   core;
    {
        std::lock_guard<std::mutex> lock(g_stub_mutex);

        auto it = g_ctxs.find(ctx);
        if (it == g_ctxs.end()) return RKNN_ERR_CTX_INVALID;
        stub_ctx & sc = it->second;
        if (!sc.A || !sc.B || !sc.C) return RKNN_ERR_PARAM_INVALID;

        M = sc.info.M; K = sc.info.K; N = sc.info.N;
        core = sc.core;

        // the device reads its own copy, not whatever the CPU last wrote
        A = (const uint16_t *) g_mems.find(sc.A)->second->device.data();
        B = (const uint16_t *) g_mems.find(sc.B)->second->device.data();
        C = (float *)          g_mems.find(sc.C)->second->device.data();
    }

    const stub_soc soc = stub_get_soc();

    const int inflight = ++g_stub_inflight;
    int prev = g_stub_max_inflight.load();
    while (inflight > prev && !g_stub_max_inflight.compare_exchange_weak(prev, inflight)) {}

    // optional delay so concurrent runs really overlap in time
    if (const char * d = getenv("STUB_RUN_DELAY_MS")) {
        std::this_thread::sleep_for(std::chrono::milliseconds(atoi(d)));
    }

    // decode both operands once, B straight out of the native layout, so a real model runs
    // through the stub in seconds instead of minutes (end-to-end tests)
    std::vector<float> a((size_t) M * K), bt((size_t) N * K);
    for (size_t i = 0; i < a.size(); i++) {
        a[i] = fp16_to_fp32(A[i]);
    }
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            bt[(size_t) n * K + k] = fp16_to_fp32(B[native_index(k, n, K, soc.ka, soc.na)]);
        }
    }
    for (int m = 0; m < M; m++) {
        const float * am = a.data() + (size_t) m * K;
        for (int n = 0; n < N; n++) {
            const float * bn = bt.data() + (size_t) n * K;
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += am[k] * bn[k];
            }
            C[(size_t) m * N + n] = acc;
        }
    }
    g_stub_inflight--;
    g_stub_runs++;
    if (M > 1) {
        g_stub_work_runs++;
    }
    g_stub_core_mask |= (int) core;
    return RKNN_SUCC;
}

int rknn_matmul_destroy(rknn_matmul_ctx ctx) {
    std::lock_guard<std::mutex> lock(g_stub_mutex);
    g_ctxs.erase(ctx);
    return RKNN_SUCC;
}

int rknn_B_normal_layout_to_native_layout(void * B_input, void * B_output, int K, int N, rknn_matmul_info * info) {
    const stub_soc soc = stub_get_soc();

    // fault injection: weight conversion fails, which forces the CPU fallback
    if (const char * fk = getenv("STUB_FAIL_CONVERT_K")) {
        if (K == atoi(fk)) return RKNN_ERR_FAIL;
    }

    if (info->K != K || info->N != N) return RKNN_ERR_PARAM_INVALID;
    if (K % soc.ka != 0 || N % soc.na != 0) return RKNN_ERR_PARAM_INVALID;

    // real hardware would write a segmented layout here and the matmul would be silently wrong
    if (soc.ksplit > 0 && K > soc.ksplit) {
        fprintf(stderr, "stub: B conversion with K=%d above the %s split threshold %d\n", K, soc.name, soc.ksplit);
        abort();
    }

    const uint16_t * in  = (const uint16_t *) B_input;
    uint16_t *       out = (uint16_t *) B_output;
    for (int k = 0; k < K; k++) {
        for (int n = 0; n < N; n++) {
            out[native_index(k, n, K, soc.ka, soc.na)] = in[(size_t) k * N + n];
        }
    }
    return RKNN_SUCC;
}

rknn_tensor_mem * rknn_create_mem(rknn_context, uint32_t size) {
    std::lock_guard<std::mutex> lock(g_stub_mutex);
    rknn_tensor_mem * mem = (rknn_tensor_mem *) calloc(1, sizeof(rknn_tensor_mem));
    stub_mem * sm = new stub_mem;
    sm->host.assign(size, 0);
    sm->device.assign(size, 0);
    mem->virt_addr = sm->host.data();
    mem->size = size;
    g_mems[mem] = sm;
    return mem;
}

int rknn_mem_sync(rknn_context, rknn_tensor_mem * mem, rknn_mem_sync_mode mode) {
    std::lock_guard<std::mutex> lock(g_stub_mutex);
    auto it = g_mems.find(mem);
    if (it == g_mems.end()) return RKNN_ERR_PARAM_INVALID;
    if (mode & RKNN_MEMORY_SYNC_TO_DEVICE) {
        it->second->device = it->second->host;
    }
    if (mode & RKNN_MEMORY_SYNC_FROM_DEVICE) {
        it->second->host = it->second->device;
    }
    return RKNN_SUCC;
}

int rknn_destroy_mem(rknn_context, rknn_tensor_mem * mem) {
    std::lock_guard<std::mutex> lock(g_stub_mutex);
    auto it = g_mems.find(mem);
    if (it != g_mems.end()) {
        delete it->second;
        g_mems.erase(it);
    }
    free(mem);
    return RKNN_SUCC;
}

int rknn_query(rknn_context, rknn_query_cmd cmd, void * info, uint32_t size) {
    if (cmd == RKNN_QUERY_SDK_VERSION && size >= sizeof(rknn_sdk_version)) {
        rknn_sdk_version * v = (rknn_sdk_version *) info;
        strcpy(v->api_version, "2.3.0 (stub)");
        strcpy(v->drv_version, "0.9.8 (stub)");
        return RKNN_SUCC;
    }
    return RKNN_ERR_PARAM_INVALID;
}

} // extern "C"
