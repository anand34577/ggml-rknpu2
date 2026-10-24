// Empirical test: does rknn_matmul_set_core_mask do anything useful on this chip?
// The header says the matmul-level core mask is RK3588 only. Rather than trust that
// comment, ask the real runtime: try every mask, see which ones a matmul context
// accepts, and time a large matmul under each to see if throughput actually changes.
//
// build:  g++ -O2 -o probe-cores probe-cores.cpp -lrknnrt
// run:    ./probe-cores

#include "rknn_api.h"
#include "rknn_matmul_api.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

static double now_ms(void) {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

struct mask_entry { rknn_core_mask mask; const char * name; };

static const mask_entry masks[] = {
    { RKNN_NPU_CORE_AUTO,    "AUTO" },
    { RKNN_NPU_CORE_0,       "CORE_0" },
    { RKNN_NPU_CORE_1,       "CORE_1" },
    { RKNN_NPU_CORE_2,       "CORE_2" },
    { RKNN_NPU_CORE_0_1,     "CORE_0_1" },
    { RKNN_NPU_CORE_0_1_2,   "CORE_0_1_2" },
    { RKNN_NPU_CORE_ALL,     "CORE_ALL" },
};

int main(void) {
    rknn_sdk_version ver;
    memset(&ver, 0, sizeof(ver));

    // large, well aligned fp16 matmul so a real throughput difference is visible
    const int M = 512, K = 4096, N = 4096;

    printf("M=%d K=%d N=%d, each mask: create -> set_core_mask -> 5 timed runs\n\n", M, K, N);
    printf("%-12s %-14s %-14s %10s\n", "mask", "create()", "set_core_mask()", "avg ms");
    printf("--------------------------------------------------------------\n");

    for (const mask_entry & me : masks) {
        rknn_matmul_ctx     ctx;
        rknn_matmul_info    info;
        rknn_matmul_io_attr io_attr;
        memset(&info, 0, sizeof(info));
        info.M = M; info.K = K; info.N = N;
        info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
        info.B_layout = RKNN_MM_LAYOUT_NATIVE;
        info.AC_layout = RKNN_MM_LAYOUT_NORM;

        int rc_create = rknn_matmul_create(&ctx, &info, &io_attr);
        if (rc_create != RKNN_SUCC) {
            printf("%-12s %-14d %-14s %10s\n", me.name, rc_create, "-", "-");
            continue;
        }

        int rc_mask = rknn_matmul_set_core_mask(ctx, me.mask);

        rknn_tensor_mem * A = rknn_create_mem(ctx, io_attr.A.size);
        rknn_tensor_mem * B = rknn_create_mem(ctx, io_attr.B.size);
        rknn_tensor_mem * C = rknn_create_mem(ctx, io_attr.C.size);
        rknn_matmul_set_io_mem(ctx, A, &io_attr.A);
        rknn_matmul_set_io_mem(ctx, B, &io_attr.B);
        rknn_matmul_set_io_mem(ctx, C, &io_attr.C);
        memset(A->virt_addr, 0x3c, io_attr.A.size); // ~1.0 in fp16, avoids NaN/inf noise
        memset(B->virt_addr, 0x3c, io_attr.B.size);
        rknn_mem_sync(ctx, A, RKNN_MEMORY_SYNC_TO_DEVICE);
        rknn_mem_sync(ctx, B, RKNN_MEMORY_SYNC_TO_DEVICE);

        rknn_matmul_run(ctx); // warm up

        const double t0 = now_ms();
        for (int i = 0; i < 5; i++) {
            rknn_matmul_run(ctx);
        }
        const double avg = (now_ms() - t0) / 5.0;

        printf("%-12s %-14d %-14d %10.2f\n", me.name, rc_create, rc_mask, avg);

        rknn_destroy_mem(ctx, A);
        rknn_destroy_mem(ctx, B);
        rknn_destroy_mem(ctx, C);
        rknn_matmul_destroy(ctx);
    }

    printf("\nread it as: create()/set_core_mask() == 0 means accepted, nonzero means the\n");
    printf("runtime rejected that mask. If every accepted mask times about the same, the\n");
    printf("mask is a no-op on this chip even though it 'succeeds'. If CORE_0/1/2 succeed\n");
    printf("and CORE_0_1 or CORE_0_1_2 is meaningfully faster than a single core, real\n");
    printf("multi-core matmul is available. The backend tests pinning itself at startup and\n");
    printf("logs 'cores N (pinned)' or 'cores N (parallel)' for the mode it picked.\n");

    return 0;
}
