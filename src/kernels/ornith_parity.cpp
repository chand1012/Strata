// Ornith's wider CPU pool, mixed projections, and CUDA compatibility path vs ggml.
#include "ggml-cpu.h"
#include "ggml.h"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>
namespace K = strata::kernels;
namespace C = strata::kernels::cpu;
static void ck(cudaError_t e) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(e));
        std::exit(2);
    }
}
template <class F> static float timed(F run) {
    run();
    ck(cudaDeviceSynchronize());
    cudaEvent_t begin, end;
    ck(cudaEventCreate(&begin));
    ck(cudaEventCreate(&end));
    ck(cudaEventRecord(begin));
    for (int i = 0; i < 20; ++i)
        run();
    ck(cudaEventRecord(end));
    ck(cudaEventSynchronize(end));
    float ms = 0;
    ck(cudaEventElapsedTime(&ms, begin, end));
    ck(cudaEventDestroy(begin));
    ck(cudaEventDestroy(end));
    return ms / 20;
}
template <class T> T *dev(size_t n) {
    T *p;
    ck(cudaMalloc(&p, n * sizeof(T)));
    ck(cudaMemset(p, 0, n * sizeof(T)));
    return p;
}
template <class T> void put(T *d, const T *h, size_t n) { ck(cudaMemcpy(d, h, n * sizeof(T), cudaMemcpyHostToDevice)); }
template <class T> void get(T *h, const T *d, size_t n) { ck(cudaMemcpy(h, d, n * sizeof(T), cudaMemcpyDeviceToHost)); }
static void compare(const char *label, const std::vector<float> &got, const std::vector<float> &ref, float tol) {
    double err = 0, norm = 0;
    float maxerr = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i]))
            std::exit(3);
        double d = got[i] - ref[i];
        err += d * d;
        norm += (double)ref[i] * ref[i];
        maxerr = std::max(maxerr, (float)std::abs(d));
    }
    double rel = std::sqrt(err / std::max(norm, 1e-30));
    std::printf("%s relative L2 %.8g max %.8g\n", label, rel, maxerr);
    if (rel > tol)
        std::exit(4);
}
static std::vector<uint8_t> weights(int type, int n, int rows) {
    std::vector<float> f((size_t)n * rows), importance(n, 1.f);
    for (size_t i = 0; i < f.size(); ++i)
        f[i] = 0.005f * std::sin(float(i) * 0.173f) + 0.004f * std::cos(float(i) * 0.031f);
    std::vector<uint8_t> q(ggml_row_size((ggml_type)type, n) * rows);
    ggml_quantize_chunk((ggml_type)type, f.data(), q.data(), 0, rows, n, importance.data());
    return q;
}
// q8_1 wire format: fp16 d, fp16 sum, 32 int8 values.
static std::vector<float> q8float(const std::vector<uint8_t> &q, int n) {
    std::vector<float> f(n);
    for (int i = 0; i < n; ++i) {
        uint16_t h;
        std::memcpy(&h, q.data() + (i / 32) * 36, 2);
        f[i] = ggml_fp16_to_fp32(h) * (int8_t)q[(i / 32) * 36 + 4 + i % 32];
    }
    return f;
}
static std::vector<float> project(int type, const uint8_t *w, int n, int rows, const std::vector<float> &x) {
    std::vector<float> y(rows), row(n);
    size_t rb = ggml_row_size((ggml_type)type, n);
    for (int r = 0; r < rows; ++r) {
        ggml_get_type_traits((ggml_type)type)->to_float(w + r * rb, row.data(), n);
        double s = 0;
        for (int i = 0; i < n; ++i)
            s += (double)row[i] * x[i];
        y[r] = (float)s;
    }
    return y;
}
// IQ1_S integer dots apply the asymmetric offset to Q8_1's stored raw
// activation sum. The dequantize-and-multiply path instead sums reconstructed
// int8 activations. Account for that documented wire-format difference in the
// independent scalar reference, without relaxing the numerical tolerance.
static void iq1_sum_reference(std::vector<float> &y, int type, const uint8_t *w, int n, const std::vector<uint8_t> &q) {
    if (type != 19)
        return;
    const auto x = q8float(q, n);
    const size_t rb = ggml_row_size((ggml_type)type, n);
    for (size_t r = 0; r < y.size(); ++r) {
        double correction = 0;
        for (int b = 0; b < n / 256; ++b) {
            const uint8_t *block = w + r * rb + b * 50;
            uint16_t dh;
            std::memcpy(&dh, block, 2);
            for (int j = 0; j < 8; ++j) {
                uint16_t qh, sh;
                std::memcpy(&qh, block + 34 + j * 2, 2);
                std::memcpy(&sh, q.data() + (b * 8 + j) * 36 + 2, 2);
                double reconstructed = 0;
                for (int k = 0; k < 32; ++k)
                    reconstructed += x[b * 256 + j * 32 + k];
                const float scale = ggml_fp16_to_fp32(dh) * (((qh >> 11) & 14) + 1);
                const float offset = (qh & 0x8000) ? -1.125f : -0.875f;
                correction += scale * offset * (ggml_fp16_to_fp32(sh) - reconstructed);
            }
        }
        y[r] += (float)correction;
    }
}

int main(int argc, char **argv) {
    FILE *dump = argc == 2 ? std::fopen(argv[1], "wb") : nullptr;
    if (argc == 2 && !dump)
        return 7;
    ggml_cpu_init();
    constexpr int N = 3072, F = 1024;
    for (auto types : {std::vector<int>{19, 19, 10},
                       {11, 11, 11},
                       {12, 13, 13},
                       {18, 18, 22},
                       {21, 21, 18},
                       {16, 16, 10},
                       {22, 22, 18},
                       {18, 18, 23},
                       {8, 8, 8}}) {
        for (int type : types) {
            int elems, bytes;
            if (!strata::block_geometry(type, elems, bytes) || elems != ggml_blck_size((ggml_type)type) ||
                bytes != (int)ggml_type_size((ggml_type)type))
                return 6;
        }
        int gt = types[0], ut = types[1], dt = types[2];
        std::printf("types %d/%d/%d\n", gt, ut, dt);
        C::NativeFmt f;
        std::string err;
        if (!C::native_fmt_mixed(gt, ut, dt, N, F, f, err)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        auto g = weights(gt, N, F), u = weights(ut, N, F), d = weights(dt, F, N);
        std::vector<uint8_t> blob(f.bytes);
        std::memcpy(blob.data(), g.data(), g.size());
        std::memcpy(blob.data() + f.up_off, u.data(), u.size());
        std::memcpy(blob.data() + f.down_off, d.data(), d.size());
        std::vector<float> x(N);
        for (int i = 0; i < N; ++i)
            x[i] = std::sin(float(i) * 0.031f);
        std::vector<uint8_t> act(C::kNativeActBytes), hq(C::kNativeHBytes);
        C::native_quant_act(f, x.data(), act.data());
        std::vector<float> h(F), ref(N), out(N), out2(N);
        // Independent ggml vec_dot calls, not native_gu_rows, catch mixed
        // offsets/types.
        for (int r = 0; r < F; ++r) {
            float a, b;
            ggml_get_type_traits_cpu((ggml_type)gt)->vec_dot(N, &a, 0, blob.data() + r * f.gu_row, 0, act.data(), 0, 1);
            ggml_get_type_traits_cpu((ggml_type)ut)
                ->vec_dot(N, &b, 0, blob.data() + f.up_off + r * ggml_row_size((ggml_type)ut, N), 0, act.data(), 0, 1);
            h[r] = a / (1 + std::exp(-a)) * b;
        }
        C::native_quant_h(f, h.data(), hq.data());
        for (int r = 0; r < N; ++r)
            ggml_get_type_traits_cpu((ggml_type)dt)
                ->vec_dot(F, &ref[r], 0, blob.data() + f.down_off + r * f.d_row, 0, hq.data(), 0, 1);
        C::ExpertJobMulti job;
        job.blob = blob.data();
        job.nt = 2;
        job.nact[0] = act.data();
        job.out[0] = out.data();
        job.nact[1] = act.data();
        job.out[1] = out2.data();
        {
            C::ExpertPool pool(3, false);
            pool.run_split_multi_native(f, &job, 1);
        }
        compare("CPU pool token 0", out, ref, 2e-5f);
        compare("CPU pool token 1", out2, ref, 2e-5f);
        auto L = K::native_expert_layout(gt, dt, N, F);
        L.compatibility = true;
        L.u_type = ut;
        L.u_row = ggml_row_size((ggml_type)ut, N);
        L.down_off = f.down_off;
        L.bytes = f.bytes;
        auto wb = dev<uint8_t>(blob.size());
        put(wb, blob.data(), blob.size());
        auto xd = dev<float>(N);
        put(xd, x.data(), x.size());
        auto xq = dev<uint8_t>((N / 32) * 36);
        K::quantize_q8_1_rows(xd, 1, N, xq, nullptr);
        // Each newly used decoder must match ggml before checking matrix
        // arithmetic.
        for (int t : types) {
            auto w = weights(t, 256, 1);
            auto wd = dev<uint8_t>(w.size());
            put(wd, w.data(), w.size());
            auto fd = dev<float>(256);
            K::iq_dequant_f32(t, wd, 256, fd, nullptr);
            std::vector<float> a(256), b(256);
            get(a.data(), fd, 256);
            ggml_get_type_traits((ggml_type)t)->to_float(w.data(), b.data(), 256);
            compare("dequant", a, b, 1e-6f);
            ck(cudaFree(wd));
            ck(cudaFree(fd));
        }
        auto ptr = dev<unsigned long long>(1);
        unsigned long long addr = (unsigned long long)wb;
        put(ptr, &addr, 1);
        auto start = dev<int32_t>(2);
        int32_t bounds[] = {0, 1};
        put(start, bounds, 2);
        auto ng = dev<int32_t>(1);
        int32_t one = 1;
        put(ng, &one, 1);
        auto ids = dev<int32_t>(1);
        auto scratch = dev<uint8_t>(K::native_expert_scratch_bytes(1, F));
        auto yd = dev<float>(N);
        const float kernel_ms =
            timed([&] { K::native_expert_grouped(L, ptr, start, ng, ids, ids, 1, 1, xq, scratch, yd, nullptr); });
        std::printf("ORNITH_KERNEL types=%d/%d/%d entries=1 ms=%.6f\n", gt, ut, dt, kernel_ms);
        ck(cudaDeviceSynchronize());
        get(out.data(), yd, N);
        if (dump)
            std::fwrite(out.data(), sizeof(float), N, dump);
        std::vector<uint8_t> qa((N / 32) * 36), qh((F / 32) * 36);
        get(qa.data(), xq, qa.size());
        get(qh.data(), scratch + 3 * F * sizeof(float), qh.size());
        auto xf = q8float(qa, N);
        auto gr = project(gt, g.data(), N, F, xf), ur = project(ut, u.data(), N, F, xf);
        std::vector<float> gg(F), uu(F);
        get(gg.data(), (float *)scratch, F);
        get(uu.data(), (float *)(scratch + F * sizeof(float)), F);
        const char *mode = std::getenv("STRATA_ORNITH_DECODE");
        if (mode && std::atoi(mode) == 3) {
            iq1_sum_reference(gr, gt, g.data(), N, qa);
            iq1_sum_reference(ur, ut, u.data(), N, qa);
        }
        compare("GPU gate", gg, gr, 2e-5f);
        compare("GPU up", uu, ur, 2e-5f);
        auto dr = project(dt, d.data(), F, N, q8float(qh, F));
        if (mode && std::atoi(mode) == 3)
            iq1_sum_reference(dr, dt, d.data(), F, qh);
        compare("GPU down", out, dr, 2e-5f);
        // Multiple tokens per group exercise per-entry scratch reuse.
        // Two experts (the second has a zero gate), with permuted
        // source/destination rows.
        {
            constexpr int count = 3;
            auto zero_blob = blob;
            std::vector<float> zeros(N, 0.0f), importance(N, 1.0f);
            for (int row = 0; row < F; ++row)
                ggml_quantize_chunk((ggml_type)gt, zeros.data(), zero_blob.data() + row * f.gu_row, 0, 1, N,
                                    importance.data());
            auto wb2 = dev<uint8_t>(zero_blob.size());
            put(wb2, zero_blob.data(), zero_blob.size());
            auto xp = dev<float>(count * N);
            for (int t = 0; t < count; ++t)
                put(xp + t * N, x.data(), N);
            auto qp = dev<uint8_t>(count * qa.size());
            K::quantize_q8_1_rows(xp, count, N, qp, nullptr);
            auto pp = dev<unsigned long long>(2);
            unsigned long long ptrs[] = {(unsigned long long)wb, (unsigned long long)wb2};
            put(pp, ptrs, 2);
            auto sp = dev<int32_t>(3);
            int32_t starts[] = {0, 2, 3};
            put(sp, starts, 3);
            auto np = dev<int32_t>(1);
            int32_t two = 2;
            put(np, &two, 1);
            auto dp = dev<int32_t>(3);
            int32_t dest[] = {2, 1, 0};
            put(dp, dest, 3);
            auto tp = dev<int32_t>(3);
            int32_t toks[] = {1, 0, 2};
            put(tp, toks, 3);
            auto sc = dev<uint8_t>(K::native_expert_scratch_bytes(count, F));
            auto yp = dev<float>(count * N);
            K::native_expert_grouped(L, pp, sp, np, dp, tp, 2, count, qp, sc, yp, nullptr);
            std::vector<float> got(count * N), expected(count * N, 0.0f);
            get(got.data(), yp, got.size());
            if (dump)
                std::fwrite(got.data(), sizeof(float), got.size(), dump);
            std::copy(out.begin(), out.end(), expected.begin() + N);
            std::copy(out.begin(), out.end(), expected.begin() + 2 * N);
            compare("GPU grouped multi-token", got, expected, 2e-5f);
            for (void *p : {(void *)wb2, (void *)xp, (void *)qp, (void *)pp, (void *)sp, (void *)np, (void *)dp,
                            (void *)tp, (void *)sc, (void *)yp})
                ck(cudaFree(p));
        }
        // Empty groups must not touch existing output.
        int32_t zero = 0;
        put(ng, &zero, 1);
        K::native_expert_grouped(L, ptr, start, ng, ids, ids, 1, 1, xq, scratch, yd, nullptr);
        std::vector<float> empty(N);
        get(empty.data(), yd, N);
        if (std::memcmp(empty.data(), out.data(), N * sizeof(float)))
            return 5;
        for (void *p : {(void *)wb, (void *)xd, (void *)xq, (void *)ptr, (void *)start, (void *)ng, (void *)ids,
                        (void *)scratch, (void *)yd})
            ck(cudaFree(p));
    }
    if (dump)
        std::fclose(dump);
    std::puts("Ornith expert parity passed");
}
