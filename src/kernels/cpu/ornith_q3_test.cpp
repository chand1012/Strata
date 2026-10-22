// Q3_K and singleton IQ CPU rows at Ornith's projection widths. No GPU required.
#include "strata/kernels/cpu/kq_avx2.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <vector>
#include <random>
#include <cstring>
#include <cstdio>
#include <chrono>
#include <cmath>
#include <cstdlib>

int main() {
    namespace C = strata::kernels::cpu;
    if (!C::cpu_avx2_ok() || !ggml_cpu_has_avx2() || ggml_cpu_has_avx512()) {
        std::puts("SKIP: exact ggml AVX2 reference required"); return 77;
    }
    ggml_cpu_init();
    std::mt19937 rng(724);
    std::normal_distribution<float> normal(0,0.2f);
    int failures = 0;
    for (int n : {1024,3072}) {
        constexpr int rows = 35;
        const size_t wr = ggml_row_size(GGML_TYPE_Q3_K,n), ar = ggml_row_size(GGML_TYPE_Q8_K,n);
        const auto* wt = ggml_get_type_traits(GGML_TYPE_Q3_K);
        const auto* at = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
        const auto dot = ggml_get_type_traits_cpu(GGML_TYPE_Q3_K)->vec_dot;
        std::vector<unsigned char> w(2*rows*wr), a(8*ar);
        std::vector<float> x(n), got(8*rows), ref(8*rows), gu(8*rows);
        const void* ap[8]; float* op[8]; float* gp[8];
        for (int r=0;r<2*rows;++r) {
            for (float& f:x) f=normal(rng);
            wt->from_float_ref(x.data(),w.data()+r*wr,n);
        }
        for (int t=0;t<8;++t) {
            for (float& f:x) f=t == 7 ? 0 : normal(rng);
            at->from_float(x.data(),a.data()+t*ar,n);
            ap[t]=a.data()+t*ar;op[t]=got.data()+t*rows;gp[t]=gu.data()+t*rows;
        }
        for (int nt=1;nt<=8;++nt) {
            std::fill(got.begin(),got.end(),-100.f);
            C::kq256_rows(11,w.data(),wr,n,ap,nt,op,1,rows-1);
            C::kq256_gu_rows(11,w.data(),wr,rows*wr,n,ap,nt,gp,1,rows-1);
            for (int t=0;t<nt;++t) {
                if (op[t][0]!=-100.f || op[t][rows-1]!=-100.f) ++failures;
                for (int r=1;r<rows-1;++r) {
                    float g,u;
                    dot(n,&g,0,w.data()+r*wr,0,ap[t],0,1);
                    dot(n,&u,0,w.data()+(r+rows)*wr,0,ap[t],0,1);
                    float f=(g/(1.f+std::exp(-g)))*u;
                    if (std::memcmp(&g,&op[t][r],4) || std::memcmp(&f,&gp[t][r],4)) ++failures;
                }
            }
        }
        // Same rows/activations for both arms, checksummed so the work is observable.
        for (int nt : {1,2,4,8}) {
            double elapsed[2]; volatile float checksum=0;
            for(int arm=0;arm<2;++arm) {
                auto start=std::chrono::steady_clock::now();
                for(int rep=0;rep<100;++rep) {
                    if(arm) C::kq256_rows(11,w.data(),wr,n,ap,nt,op,0,rows);
                    else for(int t=0;t<nt;++t) for(int r=0;r<rows;++r) dot(n,op[t]+r,0,w.data()+r*wr,0,ap[t],0,1);
                    checksum+=op[nt-1][rep%rows];
                }
                elapsed[arm]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            }
            std::printf("n=%d nt=%d ggml_ms=%.4f candidate_ms=%.4f checksum=%g\n",n,nt,elapsed[0],elapsed[1],(double)checksum);
        }
    }
    for (int type : {16,17,18,21,22}) for (int n : {1024,3072}) {
        const size_t stride=ggml_row_size((ggml_type)type,n), ar=ggml_row_size(GGML_TYPE_Q8_K,n);
        const size_t block=ggml_type_size((ggml_type)type);
        std::vector<unsigned char> w(17*stride),a(ar);
        for(auto& b:w) b=(unsigned char)rng();
        for(size_t off=0;off<w.size();off+=block) {w[off]=0;w[off+1]=0x2c;}
        std::vector<float> x(n),got(17);
        for(float& v:x) v=normal(rng);
        ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(x.data(),a.data(),n);
        C::iq256_rows_exact_one(type,w.data(),stride,n,a.data(),got.data(),0,17);
        for(int row=0;row<17;++row) {
            float ref;
            ggml_get_type_traits_cpu((ggml_type)type)->vec_dot(n,&ref,0,w.data()+row*stride,0,a.data(),0,1);
            if(std::memcmp(&ref,&got[row],4)) ++failures;
        }
        // The opt-in singleton path must match the per-token fallback in verification groups.
        const char* mt = std::getenv("STRATA_IQ_MT_MIN");
        const char* exact = std::getenv("STRATA_ORNITH_CPU_IQ_DOWN");
        if (mt && std::strcmp(mt,"1") == 0 && exact && std::strcmp(exact,"1") == 0) {
            C::NativeFmt fmt;
            fmt.d_type=type; fmt.d_act=GGML_TYPE_Q8_K; fmt.n_ff=n; fmt.n_embd=17; fmt.d_row=stride;
            const void* act[2]={a.data(),a.data()};
            std::vector<float> one(17), two(34);
            float* single[1]={one.data()}; float* pair[2]={two.data(),two.data()+17};
            C::native_down_rows(fmt,w.data(),act,1,single,0,17);
            C::native_down_rows(fmt,w.data(),act,2,pair,0,17);
            for (int row=0;row<17;++row)
                if (std::memcmp(&one[row],&two[row],4) || std::memcmp(&one[row],&two[row+17],4)) ++failures;
        }
    }
    std::printf("Q3_K rows/SwiGLU and IQ down exact failures: %d\n",failures);
    return failures ? 1 : 0;
}
