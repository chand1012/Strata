// Ornith dimensions, independent projection offsets, padded expert strides,
// empty groups and row permutations.
#include "ggml-cpu.h"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace mmq = strata::prefill::mmq;

void ck(cudaError_t e, const char *what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

struct Dev {
    void *p = nullptr;
    explicit Dev(size_t n) { ck(cudaMalloc(&p, n), "cudaMalloc"); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev &) = delete;
    Dev &operator=(const Dev &) = delete;
};

std::vector<uint8_t> make_matrix(ggml_type t, int64_t rows, int64_t cols, int expert, int trial) {
    const auto *tr = ggml_get_type_traits(t);
    if (tr == nullptr || tr->to_float == nullptr)
        throw std::runtime_error("no reference quantizer");
    const size_t rb = ggml_row_size(t, cols);
    std::vector<uint8_t> out((size_t)rows * rb);
    std::vector<float> row((size_t)cols);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t k = 0; k < cols; ++k) {
            const int code = (int)((k * 37 + r * 19 + expert * 23 + trial * 11) % 101) - 50;
            row[(size_t)k] = 0.025f * std::sin(0.013f * (float)(k + 1) + 0.17f * (float)r + 0.31f * (float)expert) +
                             0.0015f * (float)code;
        }
        std::vector<float> importance((size_t)cols, 1.0f);
        ggml_quantize_chunk(t, row.data(), out.data() + (size_t)r * rb, 0, 1, cols, importance.data());
    }
    return out;
}

std::vector<float> make_x(int rows, int64_t cols, int trial) {
    std::vector<float> x((size_t)rows * (size_t)cols);
    for (int r = 0; r < rows; ++r)
        for (int64_t k = 0; k < cols; ++k)
            // Exactly representable Q8 activations isolate matrix/layout errors
            // from activation quantization. Each 32-value block spans +/-127.
            x[(size_t)r * (size_t)cols + (size_t)k] =
                (k % 32 == 0   ? 127
                 : k % 32 == 1 ? -127
                               : (int)((k * 73 + r * 37 + trial * 19) % 255) - 127) /
                128.0f;
    std::fill(x.end() - cols, x.end(), 0.0f); // an all-zero row: exactly zero out
    return x;
}

std::vector<int32_t> perm(int n, int shift, bool reverse) {
    std::vector<int32_t> p((size_t)n);
    for (int i = 0; i < n; ++i)
        p[(size_t)i] = (int32_t)(((reverse ? n - 1 - i : i) + shift) % n);
    return p;
}

// one grouped product: experts side by side at one stride (a zeroed tail after
// them, as prefill.cpp stages them)
void product(mmq::Context &ctx, cudaStream_t s, const char *name, ggml_type t, int64_t out_rows, int64_t cols,
             const std::vector<int> &counts, int trial) {
    if (!mmq::supported((int)t))
        throw std::runtime_error(std::string(name) + ": not covered by this build");
    const int n = (int)counts.size(), rows = std::accumulate(counts.begin(), counts.end(), 0);
    std::vector<int32_t> bounds((size_t)n + 1, 0);
    for (int e = 0; e < n; ++e)
        bounds[(size_t)e + 1] = bounds[(size_t)e] + counts[(size_t)e];
    const auto src = perm(rows, trial + 1, false), dst = perm(rows, trial + 2, true);
    const std::vector<float> x = make_x(rows, cols, trial);
    const size_t matrix_size = mmq::matrix_bytes((int)t, out_rows, cols);
    const size_t offset = 32 * ggml_type_size(t), eb = matrix_size + 64 * ggml_type_size(t);
    if (matrix_size != (size_t)out_rows * ggml_row_size(t, cols))
        throw std::runtime_error(std::string(name) + ": matrix_bytes");
    std::vector<std::vector<uint8_t>> ws;
    std::vector<uint8_t> w((size_t)n * eb + 4096, 0);
    for (int e = 0; e < n; ++e) {
        ws.push_back(make_matrix(t, out_rows, cols, e, trial));
        std::copy(ws.back().begin(), ws.back().end(), w.begin() + (ptrdiff_t)((size_t)e * eb + offset));
    }
    const int64_t ld = 2 * out_rows; // up writes the second half of a shared gate/up output
    Dev dx(x.size() * 4), dsrc((size_t)rows * 4), ddst((size_t)rows * 4), db(bounds.size() * 4), dw(w.size()),
        dxq(mmq::q8_bytes(rows, cols)), dy((size_t)rows * (size_t)ld * 4);
    ck(cudaMemcpy(dx.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMemcpy(dsrc.p, src.data(), src.size() * 4, cudaMemcpyHostToDevice), "src");
    ck(cudaMemcpy(ddst.p, dst.data(), dst.size() * 4, cudaMemcpyHostToDevice), "dst");
    ck(cudaMemcpy(db.p, bounds.data(), bounds.size() * 4, cudaMemcpyHostToDevice), "bounds");
    ck(cudaMemcpy(dw.p, w.data(), w.size(), cudaMemcpyHostToDevice), "w");
    ck(cudaMemset(dy.p, 0xff, (size_t)rows * (size_t)ld * 4), "sentinel");
    // the copies and the sentinel above run on the legacy stream, which a
    // non-blocking stream does not wait for: a pageable cudaMemcpy/cudaMemset can
    // still be in flight when the kernels start
    ck(cudaDeviceSynchronize(), "inputs");
    mmq::quantize((const float *)dx.p, (const int32_t *)dsrc.p, dxq.p, (int)t, cols, cols, rows, s);
    mmq::Product p;
    p.w = (const uint8_t *)dw.p + offset;
    p.type = (int)t;
    p.w_rows = out_rows;
    p.w_cols = cols;
    p.expert_bytes = eb;
    p.n = n;
    p.xq = dxq.p;
    p.bounds = (const int32_t *)db.p;
    p.ids = (const int32_t *)ddst.p;
    p.total_rows = rows;
    p.max_rows = *std::max_element(counts.begin(), counts.end());
    p.dst = (float *)dy.p + out_rows;
    p.ld_dst = ld;
    ctx.run(p, s);
    ck(cudaGetLastError(), "launch");
    ck(cudaStreamSynchronize(s), "sync");
    std::vector<float> got((size_t)rows * (size_t)out_rows);
    std::vector<float> padded((size_t)rows * (size_t)ld);
    ck(cudaMemcpy(padded.data(), dy.p, padded.size() * 4, cudaMemcpyDeviceToHost), "y");
    for (int row = 0; row < rows; ++row) {
        for (int64_t col = 0; col < out_rows; ++col)
            if (!std::isnan(padded[(size_t)row * ld + col]))
                throw std::runtime_error("up overwrote gate");
        std::copy_n(padded.data() + (size_t)row * ld + out_rows, out_rows, got.data() + (size_t)row * out_rows);
    }

    // Time direct MMQ against dequantizing one expert at a time for FP16 GEMM.
    // MMQ includes activation quantization; FP16 inputs are preconverted
    // (favoring FP16). Both include their weight processing on every repeat;
    // neither includes host transfers.
    std::vector<uint16_t> x16(x.size());
    for (int r = 0; r < rows; ++r)
        for (int64_t k = 0; k < cols; ++k)
            x16[(size_t)r * cols + k] = ggml_fp32_to_fp16(x[(size_t)src[r] * cols + k]);
    Dev dx16(x16.size() * 2), dy16(got.size() * 4);
    ck(cudaMemcpy(dx16.p, x16.data(), x16.size() * 2, cudaMemcpyHostToDevice), "x16");
    strata::prefill::Gemm gemm;
    std::string err;
    if (!gemm.init(s, out_rows * cols, err))
        throw std::runtime_error(err);
    auto mmq_run = [&] {
        mmq::quantize((const float *)dx.p, (const int32_t *)dsrc.p, dxq.p, (int)t, cols, cols, rows, s);
        ctx.run(p, s);
    };
    auto fp16_run = [&] {
        for (int e = 0; e < n; ++e)
            if (counts[e]) {
                strata::kernels::iq_dequant_f16((int)t, (const uint8_t *)dw.p + (size_t)e * eb + offset,
                                                out_rows * cols, gemm.scratch(), s, nullptr);
                gemm.f16((const uint16_t *)dx16.p + (size_t)bounds[e] * cols, gemm.scratch(),
                         (float *)dy16.p + (size_t)bounds[e] * out_rows, counts[e], out_rows, cols);
            }
    };
    auto timing = [&](auto run) {
        run();
        ck(cudaStreamSynchronize(s), "warm");
        cudaEvent_t a, b;
        ck(cudaEventCreate(&a), "event");
        ck(cudaEventCreate(&b), "event");
        ck(cudaEventRecord(a, s), "start");
        for (int i = 0; i < 10; ++i)
            run();
        ck(cudaEventRecord(b, s), "end");
        ck(cudaEventSynchronize(b), "sync");
        float ms;
        ck(cudaEventElapsedTime(&ms, a, b), "elapsed");
        cudaEventDestroy(a);
        cudaEventDestroy(b);
        return ms / 10;
    };
    const float mmq_ms = timing(mmq_run), fp16_ms = timing(fp16_run);
    std::printf("TIMING %s rows=%d mmq_ms=%.5f fp16_dequant_gemm_ms=%.5f\n", name, rows, mmq_ms, fp16_ms);

    const auto *tr = ggml_get_type_traits(t);
    std::vector<float> ref(got.size(), 0.0f), wd((size_t)out_rows * (size_t)cols);
    for (int e = 0; e < n; ++e) {
        const size_t rb = ggml_row_size(t, cols);
        for (int64_t o = 0; o < out_rows; ++o)
            tr->to_float(ws[(size_t)e].data() + (size_t)o * rb, wd.data() + (size_t)o * (size_t)cols, cols);
        for (int r = bounds[(size_t)e]; r < bounds[(size_t)e + 1]; ++r) {
            const float *xr = x.data() + (size_t)src[(size_t)r] * (size_t)cols;
            for (int64_t o = 0; o < out_rows; ++o) {
                double acc = 0;
                const float *wr = wd.data() + (size_t)o * (size_t)cols;
                for (int64_t k = 0; k < cols; ++k)
                    acc += (double)wr[k] * xr[k];
                ref[(size_t)dst[(size_t)r] * (size_t)out_rows + (size_t)o] = (float)acc;
            }
        }
    }
    std::vector<float> f16got(got.size());
    ck(cudaMemcpy(f16got.data(), dy16.p, f16got.size() * 4, cudaMemcpyDeviceToHost), "fp16 y");
    double f16err = 0, f16ref = 0;
    for (int r = 0; r < rows; ++r)
        for (int64_t col = 0; col < out_rows; ++col) {
            const double expected = ref[(size_t)dst[r] * out_rows + col];
            const double value = f16got[(size_t)r * out_rows + col];
            if (!std::isfinite(value))
                throw std::runtime_error("non-finite FP16 product");
            f16err += (value - expected) * (value - expected);
            f16ref += expected * expected;
        }
    const double f16rel = std::sqrt(f16err / std::max(f16ref, 1e-30));
    std::printf("FP16_REFERENCE %s relative_l2=%.6g\n", name, f16rel);
    if (f16rel > 0.01)
        throw std::runtime_error("FP16 reference mismatch");
    double err2 = 0, ref2 = 0, max_abs = 0, zero_max = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i]))
            throw std::runtime_error(std::string(name) + ": unwritten or non-finite output");
        const double d = (double)got[i] - ref[i];
        err2 += d * d;
        ref2 += (double)ref[i] * ref[i];
        max_abs = std::max(max_abs, std::fabs(d));
    }
    for (int r = 0; r < rows; ++r)
        if (src[(size_t)r] == rows - 1) // the zero row
            for (int64_t o = 0; o < out_rows; ++o)
                zero_max =
                    std::max(zero_max, (double)std::fabs(got[(size_t)dst[(size_t)r] * (size_t)out_rows + (size_t)o]));
    const double rms = std::sqrt(ref2 / (double)ref.size()), rel = std::sqrt(err2 / (double)ref.size()) / rms;
    std::printf("%-22s rows %2d experts %d  ref_rms %.4g  rel_l2 %.5f  max/rms "
                "%.4f  zero row %.3g\n",
                name, rows, n, rms, rel, max_abs / rms, zero_max);
    if (rel > 0.005 || max_abs / rms > 0.02 || zero_max > std::max(1e-5, 1e-4 * rms))
        throw std::runtime_error(std::string(name) + ": outside the screen");
}
} // namespace

int main() {
    try {
        ggml_cpu_init();
        if (!mmq::built()) {
            std::printf("no MMQ in this build\n");
            return 1;
        }
        cudaStream_t s = nullptr;
        ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
        {
            mmq::Context ctx;
            const std::vector<std::vector<int>> batches{{1, 0, 3}, {8, 0, 17}, {32, 1, 64}, {128}};
            int trial = 0;
            for (const auto &counts : batches) {
                for (ggml_type type : {GGML_TYPE_IQ1_S, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K,
                                       GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_S,
                                       GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_0}) {
                    const std::string tag = std::string(ggml_type_name(type)) + "-" + std::to_string(trial);
                    product(ctx, s, (tag + " gate/up").c_str(), type, 1024, 3072, counts, trial);
                    product(ctx, s, (tag + " down").c_str(), type, 3072, 1024, counts, trial + 5);
                }
                ++trial;
            }
        }
        ck(cudaStreamDestroy(s), "destroy");
        std::printf("Ornith MMQ screen passed\n");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "Ornith MMQ screen failed: %s\n", e.what());
        return 1;
    }
}
