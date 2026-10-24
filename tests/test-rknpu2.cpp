// Correctness tests for the ggml-rknpu2 backend against the strict CPU stub.
//
// Covers the mappings that are easy to get wrong: weight transpose and native
// layout, fp16 conversion, quantized weights, 3D activations flattened into one
// matmul, K splitting for wide weights, M tiling with a padded tail, cache sync,
// and the shape rules supports_op must enforce per chip.
//
// usage: test-rknpu2 [soc]      soc defaults to rk3588

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-rknpu2.h"
#include "rknn_api.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <atomic>
extern std::atomic<int> g_stub_runs;
extern std::atomic<int> g_stub_core_mask;
extern std::atomic<int> g_stub_max_inflight;

static int g_fail = 0;

static void set_env(const char * k, const char * v) {
#ifdef _WIN32
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

static float frand(uint32_t & seed) {
    seed = seed * 1664525u + 1013904223u;
    return ((seed >> 8) & 0xffff) / 65536.0f - 0.5f;
}

// reference matmul, both operands rounded to fp16 the way the NPU sees them. Activations
// fp16 cannot hold stay fp32, the backend must scale those rows instead of making inf.
static void ref_mul_mat(const std::vector<float> & w, const std::vector<float> & x,
                        std::vector<float> & out, int64_t M, int64_t K, int64_t N) {
    out.assign((size_t) M * N, 0.0f);
    for (int64_t n = 0; n < N; n++) {
        for (int64_t m = 0; m < M; m++) {
            double acc = 0.0;
            for (int64_t k = 0; k < K; k++) {
                const float wf = ggml_fp16_to_fp32(ggml_fp32_to_fp16(w[(size_t) n * K + k]));
                const float xv = x[(size_t) m * K + k];
                const float xf = fabsf(xv) < 65504.0f ? ggml_fp16_to_fp32(ggml_fp32_to_fp16(xv)) : xv;
                acc += (double) wf * xf;
            }
            out[(size_t) m * N + n] = (float) acc;
        }
    }
}

static void check(const char * name, const float * got, const std::vector<float> & want) {
    double max_err = 0.0;
    for (size_t i = 0; i < want.size(); i++) {
        const double denom = fabs(want[i]) > 1.0 ? fabs(want[i]) : 1.0;
        const double err   = fabs(got[i] - want[i]) / denom;
        if (err > max_err) max_err = err;
    }
    const bool ok = max_err < 1e-2 && std::isfinite(max_err);
    printf("  %-42s max rel err %.3e  %s\n", name, max_err, ok ? "OK" : "FAIL");
    if (!ok) g_fail++;
}

static void expect(const char * name, bool cond) {
    printf("  %-42s %s\n", name, cond ? "OK" : "FAIL");
    if (!cond) g_fail++;
}

struct test_env {
    ggml_context *        ctx;
    ggml_backend_t        backend;
    ggml_backend_dev_t    dev;
    ggml_backend_buffer_t wbuf = nullptr;
};

// build one mul_mat, run it on the NPU backend and compare against the reference
static void run_case(test_env & env, const char * name, ggml_type wtype,
                     int64_t K, int64_t N, int64_t ne1, int64_t ne2, float outlier = 0.0f) {
    const int64_t M = ne1 * ne2;

    ggml_init_params ip = { 256u * 1024 * 1024, NULL, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * w = ggml_new_tensor_2d(ctx, wtype, K, N);
    ggml_tensor * x = ne2 == 1 ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, ne1)
                               : ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, ne1, ne2);
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
    assert(buf != NULL);
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    uint32_t seed = 1234;
    std::vector<float> wf((size_t) K * N), xf((size_t) K * M);
    for (auto & v : wf) v = frand(seed);
    for (auto & v : xf) v = frand(seed);
    if (outlier != 0.0f) {
        // one huge value in a few rows, like the activation outliers some LLMs have
        for (int64_t m = 1; m < M; m += 7) {
            xf[(size_t) m * K + (m % K)] = (m % 2) ? outlier : -outlier;
        }
    }

    if (wtype == GGML_TYPE_F32) {
        memcpy(w->data, wf.data(), ggml_nbytes(w));
    } else if (wtype == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> h((size_t) K * N);
        ggml_fp32_to_fp16_row(wf.data(), h.data(), (int64_t) K * N);
        memcpy(w->data, h.data(), ggml_nbytes(w));
    } else {
        ggml_quantize_chunk(wtype, wf.data(), w->data, 0, N, K, NULL);
        // compare against what the weight actually holds after quantization
        const ggml_type_traits * tt = ggml_get_type_traits(wtype);
        for (int64_t n = 0; n < N; n++) {
            tt->to_float((const char *) w->data + n * w->nb[1], wf.data() + n * K, K);
        }
    }
    memcpy(x->data, xf.data(), ggml_nbytes(x));

    if (!ggml_backend_dev_supports_op(env.dev, y)) {
        printf("  %-42s SKIPPED (not offloaded)\n", name);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    assert(ggml_gallocr_alloc_graph(galloc, gf));

    const int runs_before = g_stub_runs;
    assert(ggml_backend_graph_compute(env.backend, gf) == GGML_STATUS_SUCCESS);
    const bool used_npu = g_stub_runs > runs_before;

    std::vector<float> want;
    ref_mul_mat(wf, xf, want, M, K, N);
    check(name, (const float *) y->data, want);
    if (!used_npu) {
        printf("      note: computed on the CPU fallback path\n");
    }

    ggml_gallocr_free(galloc);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

// A weight the NPU could not take must never be offered again: the same tensor is asked
// twice, with the fault injection removed in between. Without that the scalar fallback is
// absorbing and every later call retries the failed allocation.
static void run_reject_sticky(test_env & env, int64_t K, int64_t N, int64_t M) {
    ggml_init_params ip = { 256u * 1024 * 1024, NULL, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K, N);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
    assert(buf != NULL);
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    uint32_t seed = 99;
    std::vector<float> wf((size_t) K * N), xf((size_t) K * M);
    for (auto & v : wf) v = frand(seed);
    for (auto & v : xf) v = frand(seed);
    ggml_fp32_to_fp16_row(wf.data(), (ggml_fp16_t *) w->data, (int64_t) K * N);
    memcpy(x->data, xf.data(), ggml_nbytes(x));

    char b[32];
    snprintf(b, sizeof(b), "%d", (int) K);
    set_env("STUB_FAIL_CONVERT_K", b);

    expect("failing weight accepted up front", ggml_backend_dev_supports_op(env.dev, y));

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    assert(ggml_gallocr_alloc_graph(galloc, gf));
    assert(ggml_backend_graph_compute(env.backend, gf) == GGML_STATUS_SUCCESS);

    std::vector<float> want;
    ref_mul_mat(wf, xf, want, M, K, N);
    check("failed weight still computes correctly", (const float *) y->data, want);

    // the runtime would take it now, the backend must still refuse the same tensor
    set_env("STUB_FAIL_CONVERT_K", "0");
    const int runs_before = g_stub_runs;
    expect("failed weight declined from then on", !ggml_backend_dev_supports_op(env.dev, y));
    expect("failed weight never reaches the NPU", g_stub_runs == runs_before);

    ggml_gallocr_free(galloc);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

// The scheduler asks supports_op for every node before any of them runs. The cap must hold
// across those questions, or a model bigger than the cap is accepted whole and the overflow
// lands on the scalar fallback at compute time. Needs GGML_RKNPU2_MEM_MB=4 at backend init.
static void run_mem_cap(test_env & env) {
    const int64_t K = 1024, N = 1536, M = 64; // 3 MB of fp16 per weight, only one fits in 4 MB

    ggml_init_params ip = { 64u * 1024 * 1024, NULL, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K, N);
    ggml_tensor * w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K, N);
    ggml_tensor * x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * y1 = ggml_mul_mat(ctx, w1, x);
    ggml_tensor * y2 = ggml_mul_mat(ctx, w2, x);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
    assert(buf != NULL);
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    uint32_t seed = 7;
    std::vector<float> wf((size_t) K * N), xf((size_t) K * M);
    for (auto & v : wf) v = frand(seed);
    for (auto & v : xf) v = frand(seed);
    ggml_fp32_to_fp16_row(wf.data(), (ggml_fp16_t *) w1->data, (int64_t) K * N);
    memset(w2->data, 0, ggml_nbytes(w2));
    memcpy(x->data, xf.data(), ggml_nbytes(x));

    expect("first weight accepted under the cap",   ggml_backend_dev_supports_op(env.dev, y1));
    expect("second weight declined, cap reserved",  !ggml_backend_dev_supports_op(env.dev, y2));
    expect("first weight still accepted on re-ask", ggml_backend_dev_supports_op(env.dev, y1));

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y1);
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    assert(ggml_gallocr_alloc_graph(galloc, gf));

    const int runs_before = g_stub_runs;
    assert(ggml_backend_graph_compute(env.backend, gf) == GGML_STATUS_SUCCESS);
    expect("reserved weight runs on the NPU", g_stub_runs > runs_before);

    std::vector<float> want;
    ref_mul_mat(wf, xf, want, M, K, N);
    check("reserved weight computes correctly", (const float *) y1->data, want);

    expect("second weight still declined after upload", !ggml_backend_dev_supports_op(env.dev, y2));

    ggml_gallocr_free(galloc);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

// llama.cpp loads the model with no_alloc to estimate memory: weights carry a WEIGHTS
// buffer but no data. Those must be declined, there is nothing to cache.
static bool supported_no_data(test_env & env, int64_t K, int64_t N, int64_t M) {
    ggml_init_params ip = { 16u * 1024 * 1024, NULL, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K, N);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);

    ggml_backend_buffer_t dummy = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 0);
    ggml_backend_buffer_set_usage(dummy, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    w->buffer = dummy;

    const bool r = ggml_backend_dev_supports_op(env.dev, y);
    w->buffer = nullptr;
    ggml_backend_buffer_free(dummy);
    ggml_free(ctx);
    return r;
}

// does supports_op accept this shape
static bool supported(test_env & env, ggml_type wtype, int64_t K, int64_t N, int64_t M, bool as_weights) {
    ggml_init_params ip = { 64u * 1024 * 1024, NULL, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * w = ggml_new_tensor_2d(ctx, wtype, K, N);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
    if (as_weights) {
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    memset(w->data, 0, ggml_nbytes(w));

    const bool r = ggml_backend_dev_supports_op(env.dev, y);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return r;
}

int main(int argc, char ** argv) {
    const std::string soc = argc > 1 ? argv[1] : "rk3588";

    set_env("GGML_RKNPU2_SOC",   soc.c_str());
    set_env("GGML_RKNPU2_MIN_M", "32");
    set_env("GGML_RKNPU2_MAX_M", "64"); // small tile so M tiling and padding are exercised

    printf("== %s ==\n", soc.c_str());

    test_env env;
    env.ctx     = nullptr;
    env.backend = ggml_backend_rknpu2_init();
    assert(env.backend != NULL);
    env.dev = ggml_backend_get_device(env.backend);

    const bool is3566  = soc == "rk3566" || soc == "rk3568";
    const int  n_cores = soc == "rk3588" ? 3 : soc == "rk3576" ? 2 : 1;
    const int64_t ka = is3566 ? 16 : 32;
    const int64_t na = is3566 ?  8 : 16;

    // exact tile, padded tail, and more rows than one tile
    run_case(env, "f16 K=64 N=32 M=64 (exact tile)",   GGML_TYPE_F16, 64, 32,  64, 1);
    run_case(env, "f16 K=64 N=32 M=100 (tiled+padded)", GGML_TYPE_F16, 64, 32, 100, 1);
    run_case(env, "f16 K=64 N=32 M=33 (padded tail)",   GGML_TYPE_F16, 64, 32,  33, 1);

    // quantized weights
    run_case(env, "q4_0 K=64 N=32 M=64",  GGML_TYPE_Q4_0, 64, 32, 64, 1);
    run_case(env, "q8_0 K=64 N=32 M=64",  GGML_TYPE_Q8_0, 64, 32, 64, 1);
    run_case(env, "q4_K K=256 N=32 M=64", GGML_TYPE_Q4_K, 256, 32, 64, 1);

    // 3D activations must flatten into a single matmul (vision / batched)
    run_case(env, "f16 3D src1 ne1=5 ne2=20 (M=100)", GGML_TYPE_F16, 64, 32, 5, 20);

    // K beyond the chip limit (10240, or the B layout split threshold) must be split and accumulated
    run_case(env, "f16 K=12288 N=32 M=64 (K split)", GGML_TYPE_F16, 12288, 32, 64, 1);

    // activations past the fp16 range must not turn into inf
    run_case(env, "f16 activation outliers 1e5",          GGML_TYPE_F16, 64,    32, 64, 1, 1.0e5f);
    run_case(env, "q4_0 activation outliers 3e6 + split", GGML_TYPE_Q4_0, 12288, 32, 64, 1, 3.0e6f);

    // weight cache reuse must stay correct across calls
    run_case(env, "f16 K=64 N=32 M=64 (cached weight)", GGML_TYPE_F16, 64, 32, 64, 1);

    // K=20512 splits into 6848,6848,6816 and reuses the 6816 kernel built just below
    run_case(env, "f16 K=6816 N=32 M=32 (chunk kernel reuse)", GGML_TYPE_F16, 6816, 32, 32, 1);
    run_case(env, "f16 K=20512 N=32 M=64 (mixed tiles)",     GGML_TYPE_F16, 20512, 32, 64, 1);

    // a shape the runtime refuses after supports_op accepted it must still give the
    // right answer, and must not be offered to the NPU again
    // multi core: N is split over the cores, every core must really be used and the
    // stitched result must match the reference
    if (n_cores > 1) {
        printf("  -- multi core (%d cores, pinned) --\n", n_cores);
        g_stub_core_mask    = 0;
        g_stub_max_inflight = 0;
        set_env("STUB_RUN_DELAY_MS", "20"); // make the slice runs overlap in time
        run_case(env, "f16 N=192 M=64 (all cores)", GGML_TYPE_F16, 64, na * 12, 64, 1);
        set_env("STUB_RUN_DELAY_MS", "0");
        expect("every NPU core used, each pinned", g_stub_core_mask.load() == (1 << n_cores) - 1);
        expect("slices run at the same time",      g_stub_max_inflight.load() == n_cores);

        run_case(env, "f16 K=12288 N=96 M=64 (all cores+split)", GGML_TYPE_F16, 12288, na * 6, 64, 1);
        run_case(env, "q4_0 N=16 (too small to split)",          GGML_TYPE_Q4_0, 64, na, 64, 1);
    }

    printf("  -- runtime rejection --\n");
    {
        // K is not changed by column slicing, so it is the reliable injection key
        const int64_t fail_k = 1024; // aligned on every chip and used by no other case
        char buf[32];
        snprintf(buf, sizeof(buf), "%d", (int) fail_k);
        set_env("STUB_FAIL_K", buf);

        // a shape the runtime refuses must be refused up front, so the node goes to the
        // CPU backend and its fast kernels instead of our slow scalar fallback
        expect("refused shape never reaches compute", !supported(env, GGML_TYPE_F16, fail_k, 4 * na, 64, true));
        set_env("STUB_FAIL_K", "0");

        // a weight conversion failure can still only be caught at compute time, that path
        // must produce the right answer on the CPU
        char buf2[32];
        snprintf(buf2, sizeof(buf2), "%d", (int) (2048));
        set_env("STUB_FAIL_CONVERT_K", buf2);
        run_case(env, "conversion failure falls back correctly", GGML_TYPE_F16, 2048, 4 * na, 64, 1);
        set_env("STUB_FAIL_CONVERT_K", "0");

        run_reject_sticky(env, 4096, 4 * na, 64);
    }

    printf("  -- shape rules --\n");
    expect("aligned f16 accepted",            supported(env, GGML_TYPE_F16, 4 * ka, 4 * na, 64, true));
    expect("misaligned K rejected",           !supported(env, GGML_TYPE_F16, 4 * ka + 8, 4 * na, 64, true));
    expect("misaligned N rejected",           !supported(env, GGML_TYPE_F16, 4 * ka, 4 * na + 4, 64, true));
    expect("non weight src0 rejected",        !supported(env, GGML_TYPE_F16, 4 * ka, 4 * na, 64, false));
    expect("small batch rejected",            !supported(env, GGML_TYPE_F16, 4 * ka, 4 * na, 8, true));
    expect("f32 weights off by default",      !supported(env, GGML_TYPE_F32, 4 * ka, 4 * na, 64, true));
    expect("quantized weights accepted",      supported(env, GGML_TYPE_Q4_0, 4 * ka, 4 * na, 64, true));
    expect("weight without data rejected",    !supported_no_data(env, 4 * ka, 4 * na, 64));

    if (is3566) {
        // rk3566 has looser alignment than rk3588, the table must be used
        expect("rk3566 accepts K%16 N%8",     supported(env, GGML_TYPE_F16, 48, 24, 64, true));
    } else {
        char name[64];
        snprintf(name, sizeof(name), "%s rejects K%%16 N%%8", soc.c_str());
        expect(name,                          !supported(env, GGML_TYPE_F16, 48, 24, 64, true));
    }

    ggml_backend_free(env.backend);

    // the last backend free drops every cached weight, so a new backend starts at zero usage
    printf("  -- memory cap --\n");
    set_env("GGML_RKNPU2_MEM_MB", "4");
    env.backend = ggml_backend_rknpu2_init();
    assert(env.backend != NULL);
    run_mem_cap(env);
    ggml_backend_free(env.backend);

    // the header says the matmul core mask is RK3588 only. If the runtime refuses it, every
    // core must still be used: one unpinned context per core, run in parallel.
    if (n_cores > 1) {
        printf("  -- multi core, runtime without core mask --\n");
        set_env("STUB_NO_CORE_MASK", "1");
        env.backend = ggml_backend_rknpu2_init();
        assert(env.backend != NULL);

        g_stub_core_mask    = 0; // AUTO is 0, so any pinned run would leave bits set
        g_stub_max_inflight = 0;
        const int runs_before = g_stub_runs;
        set_env("STUB_RUN_DELAY_MS", "20");
        run_case(env, "f16 N=192 M=64 (parallel, unpinned)", GGML_TYPE_F16, 64, na * 12, 64, 1);
        set_env("STUB_RUN_DELAY_MS", "0");
        expect("one NPU run per core",            g_stub_runs - runs_before == n_cores);
        expect("no core mask used",               g_stub_core_mask.load() == 0);
        expect("unpinned slices run in parallel", g_stub_max_inflight.load() == n_cores);

        ggml_backend_free(env.backend);
        set_env("STUB_NO_CORE_MASK", "0");
    }

    // without a usable NPU the device must not be listed, or --list-devices claims an NPU
    // the scheduler will never use. Every backend is freed above, so this probes again.
    printf("  -- no NPU --\n");
    expect("NPU listed when usable", ggml_backend_reg_dev_count(ggml_backend_rknpu2_reg()) == 1);
    {
        ggml_backend_t b = ggml_backend_rknpu2_init();
        ggml_backend_free(b); // the last free resets the probe
    }
    set_env("STUB_NO_NPU", "1");
    expect("NPU not listed without driver", ggml_backend_reg_dev_count(ggml_backend_rknpu2_reg()) == 0);
    set_env("STUB_NO_NPU", "0");

    printf(g_fail == 0 ? "== %s: all passed ==\n\n" : "== %s: %d FAILED ==\n\n", soc.c_str(), g_fail);
    return g_fail == 0 ? 0 : 1;
}
