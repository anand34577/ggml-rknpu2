// Benchmark the RKNPU2 backend on real hardware.
//
// Answers the questions that decide performance and that cannot be settled off-device:
//   - how many GFLOPS the NPU actually reaches per shape
//   - whether splitting N over the cores helps (GGML_RKNPU2_CORES)
//   - which batch size is worth offloading (where the NPU passes the CPU)
//   - how much of the time is the fp32->fp16 activation conversion
//
// usage: bench-rknpu2 [K] [N]        default 4096 4096
//
// Sweep the knobs from the shell, the backend reads them once at startup:
//   for c in 1 2 3; do GGML_RKNPU2_CORES=$c ./bench-rknpu2; done     (RK3588)
//   for c in 1 2;   do GGML_RKNPU2_CORES=$c ./bench-rknpu2; done     (RK3576)

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-rknpu2.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static double now_ms(void) {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static float frand(uint32_t & seed) {
    seed = seed * 1664525u + 1013904223u;
    return ((seed >> 8) & 0xffff) / 65536.0f - 0.5f;
}

struct result { double ms; double gflops; bool offloaded; };

static result bench_one(ggml_backend_t backend, bool is_npu, ggml_type wtype,
                        int64_t K, int64_t N, int64_t M, int reps) {
    ggml_init_params ip = { (size_t) 512 * 1024 * 1024, NULL, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * w = ggml_new_tensor_2d(ctx, wtype, K, N);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    uint32_t seed = 7;
    std::vector<float> tmp((size_t) K * (N > M ? N : M));
    for (int64_t i = 0; i < K * N; i++) tmp[i] = frand(seed);
    if (wtype == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(tmp.data(), (ggml_fp16_t *) w->data, K * N);
    } else {
        ggml_quantize_chunk(wtype, tmp.data(), w->data, 0, N, K, NULL);
    }
    for (int64_t i = 0; i < K * M; i++) tmp[i] = frand(seed);
    memcpy(x->data, tmp.data(), ggml_nbytes(x));

    const bool offloaded = ggml_backend_dev_supports_op(ggml_backend_get_device(backend), y);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    ggml_gallocr_alloc_graph(galloc, gf);

    // warm up: builds the context and converts the weight, not part of the measurement
    ggml_backend_graph_compute(backend, gf);

    const double t0 = now_ms();
    for (int i = 0; i < reps; i++) {
        ggml_backend_graph_compute(backend, gf);
    }
    const double ms = (now_ms() - t0) / reps;

    ggml_gallocr_free(galloc);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    const double flop = 2.0 * (double) M * K * N;
    return { ms, flop / (ms * 1e6), is_npu ? offloaded : true };
}

int main(int argc, char ** argv) {
    const int64_t K = argc > 1 ? atoll(argv[1]) : 4096;
    const int64_t N = argc > 2 ? atoll(argv[2]) : 4096;

    ggml_backend_t npu = ggml_backend_rknpu2_init();
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (npu == NULL) {
        printf("no NPU backend\n");
        return 1;
    }
    ggml_backend_cpu_set_n_threads(cpu, 4);

    const char * cores = getenv("GGML_RKNPU2_CORES");
    printf("\nK=%lld N=%lld   GGML_RKNPU2_CORES=%s  GGML_RKNPU2_MAX_M=%s\n",
           (long long) K, (long long) N,
           cores ? cores : "(default)",
           getenv("GGML_RKNPU2_MAX_M") ? getenv("GGML_RKNPU2_MAX_M") : "(default)");

    printf("\n%-8s %-8s %12s %10s %12s %10s %8s\n", "type", "M", "NPU ms", "NPU GF/s", "CPU ms", "CPU GF/s", "speedup");
    printf("------------------------------------------------------------------------------\n");

    const ggml_type types[] = { GGML_TYPE_F16, GGML_TYPE_Q4_0 };
    const int64_t   batches[] = { 1, 32, 64, 128, 256, 512, 1024 };

    for (ggml_type t : types) {
        for (int64_t M : batches) {
            const int reps = M <= 128 ? 10 : 4;

            const result r_npu = bench_one(npu, true,  t, K, N, M, reps);
            const result r_cpu = bench_one(cpu, false, t, K, N, M, reps);

            printf("%-8s %-8lld %12.2f %10.1f %12.2f %10.1f %8.2fx%s\n",
                   ggml_type_name(t), (long long) M,
                   r_npu.ms, r_npu.gflops, r_cpu.ms, r_cpu.gflops,
                   r_cpu.ms / r_npu.ms,
                   r_npu.offloaded ? "" : "   (not offloaded, CPU numbers both sides)");
        }
    }

    // how much of the NPU time is just converting activations to fp16 on the CPU
    {
        const int64_t M = 512;
        std::vector<float>       src((size_t) M * K);
        std::vector<ggml_fp16_t> dst((size_t) M * K);
        uint32_t seed = 3;
        for (auto & v : src) v = frand(seed);

        const double t0 = now_ms();
        for (int i = 0; i < 10; i++) {
            for (int64_t m = 0; m < M; m++) {
                ggml_fp32_to_fp16_row(src.data() + m * K, dst.data() + m * K, K);
            }
        }
        printf("\nfp32->fp16 activation conversion, M=512 K=%lld: %.2f ms per matmul\n",
               (long long) K, (now_ms() - t0) / 10);
    }

    ggml_backend_free(npu);
    ggml_backend_free(cpu);
    return 0;
}
