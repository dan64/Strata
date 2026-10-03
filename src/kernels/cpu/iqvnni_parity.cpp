// src/kernels/cpu/iqvnni_parity.cpp - the AVX-VNNI kernels against their AVX2 counterparts and ggml-cpu.
//
//     build/iqvnni_parity        synthetic weights, no pack, no GPU
//
// For the pairs this CPU actually serves (IQ3_S gate/up, IQ4_NL down, Q2_0 rows): the VNNI kernel must be
// BITWISE equal to the AVX2 one - same integer values, same float order - and within 1e-5 of ggml-cpu's
// single-token vec_dot on the same quantized activations (float order only, the same contract as
// native_expert_parity).  Then the one-thread times of both, 1..4 tokens, in us and GB/s.  Exit 77
// (ctest skip) on a CPU without AVX-VNNI: there is nothing to compare there.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/iq_avxvnni.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

static constexpr int64_t H = 2560, FF = 640;
static constexpr int NT = 4;

static double rel(const std::vector<float>& a, const std::vector<float>& b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) { n += std::fabs((double) a[i] - b[i]); d += std::fabs((double) b[i]); }
    return n / (d + 1e-30);
}

// Random weights (rows of different magnitudes) quantized by ggml into the GGUF layout.
static std::vector<uint8_t> quant_rows(int type, int64_t rows, int64_t cols, int seed) {
    std::vector<float> w((size_t) (rows * cols));
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int64_t r = 0; r < rows; ++r) {
        const float scale = 0.02f * (0.5f + (float) (r % 7) / 7.0f);
        for (int64_t c = 0; c < cols; ++c) w[(size_t) (r * cols + c)] = scale * nd(rng);
    }
    std::vector<uint8_t> out((size_t) rows * ggml_row_size((ggml_type) type, cols));
    const std::vector<float> imatrix((size_t) cols, 1.0f);
    ggml_quantize_chunk((ggml_type) type, w.data(), out.data(), 0, rows, cols,
                        ggml_quantize_requires_imatrix((ggml_type) type) ? imatrix.data() : nullptr);
    return out;
}

template <typename Fn>
static double time_us(Fn&& fn, int it) {
    fn();   // warm
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < it; ++i) fn();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / it;
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    ggml_cpu_init();   // the fp16 tables and quantizers the synthesis below needs
    std::printf("iqvnni_parity: cpu %s\n", cpu::cpu_name().c_str());
    std::printf("  gates: avx2=%d avx512=%d avxvnni=%d (%s)\n", (int) cpu::cpu_avx2_ok(), (int) cpu::cpu_avx512_ok(),
                (int) cpu::cpu_avxvnni_ok(), std::getenv("STRATA_NO_VNNI") ? "STRATA_NO_VNNI set" : "no STRATA_NO_VNNI");
    if (!cpu::cpu_avx2_ok()) { std::printf("iqvnni_parity: no AVX2 - skipped\n"); return 77; }
    if (!cpu::cpu_avxvnni_ok()) { std::printf("iqvnni_parity: no AVX-VNNI on this CPU - skipped\n"); return 77; }
    int failures = 0;
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0.f, 1.f);

    // ---- (1) IQ3_S (21) gate/up rows: iqvnni vs iq256 bitwise, vs ggml-cpu, then one-thread times
    {
        const int gt = 21;
        std::vector<uint8_t> blob = quant_rows(gt, 2 * FF, H, 21);   // gate rows, then up rows
        const size_t gu_row = ggml_row_size((ggml_type) gt, H);
        const size_t up_off = gu_row * (size_t) FF;
        cpu::NativeFmt f;
        std::string err;
        if (!cpu::native_fmt(gt, 20, H, FF, f, err)) { std::printf("[21 gu] native_fmt: %s\n", err.c_str()); return 1; }
        std::vector<float> x((size_t) NT * H);
        for (auto& v : x) v = nd(rng);
        std::vector<std::vector<uint8_t>> act(NT, std::vector<uint8_t>(cpu::kNativeActBytes));
        const void* ap[NT];
        for (int t = 0; t < NT; ++t) { cpu::native_quant_act(f, x.data() + (size_t) t * H, act[t].data()); ap[t] = act[t].data(); }
        std::vector<std::vector<float>> a_out(NT, std::vector<float>(FF)), b_out(NT, std::vector<float>(FF));
        float* ao[NT]; float* bo[NT];
        for (int t = 0; t < NT; ++t) { ao[t] = a_out[t].data(); bo[t] = b_out[t].data(); }
        size_t differ = 0, values = 0;
        for (int nt = 1; nt <= NT; ++nt) {
            cpu::iq256_gu_rows(gt, blob.data(), gu_row, up_off, (int) H, ap, nt, ao, 0, (int) FF);
            cpu::iqvnni_gu_rows(gt, blob.data(), gu_row, up_off, (int) H, ap, nt, bo, 0, (int) FF);
            for (int t = 0; t < nt; ++t)
                for (int64_t r = 0; r < FF; ++r) { ++values; differ += std::memcmp(&a_out[t][r], &b_out[t][r], 4) != 0; }
        }
        std::printf("[21 gu ] iqvnni vs iq256, 1..%d tokens: %zu of %zu values differ in any bit  %s\n", NT, differ,
                    values, differ ? "FAIL" : "ok");
        if (differ) ++failures;
        {   // the gate rows against ggml-cpu's single-token vec_dot on the same Q8_K activations
            const auto* tc = ggml_get_type_traits_cpu((ggml_type) gt);
            std::vector<float> got((size_t) NT * FF), gref((size_t) NT * FF);
            float* gp[NT];
            for (int t = 0; t < NT; ++t) gp[t] = got.data() + (size_t) t * FF;
            cpu::iqvnni_rows(gt, blob.data(), gu_row, (int) H, ap, NT, gp, 0, (int) FF);
            for (int t = 0; t < NT; ++t)
                for (int64_t r = 0; r < FF; ++r)
                    tc->vec_dot((int) H, &gref[(size_t) t * FF + r], 0, blob.data() + (size_t) r * gu_row, 0, ap[t], 0, 1);
            const double rg = rel(got, gref);
            std::printf("[21 gu ] iqvnni gate rows vs ggml vec_dot: rel %.2e  %s\n", rg, rg > 1e-5 ? "FAIL" : "ok");
            if (rg > 1e-5) ++failures;
        }
        {   // the dispatcher must serve this pair with the VNNI kernel: bitwise the same answer
            std::vector<std::vector<float>> d_out(3, std::vector<float>(FF));
            float* dp[3] = {d_out[0].data(), d_out[1].data(), d_out[2].data()};
            cpu::native_gu_rows(f, blob.data(), ap, 3, dp, 0, (int) FF);
            cpu::iqvnni_gu_rows(gt, blob.data(), gu_row, up_off, (int) H, ap, 3, bo, 0, (int) FF);
            size_t d_differ = 0;
            for (int t = 0; t < 3; ++t)
                for (int64_t r = 0; r < FF; ++r) d_differ += std::memcmp(&d_out[t][r], &b_out[t][r], 4) != 0;
            std::printf("[21 gu ] dispatcher (native_gu_rows) vs iqvnni: %zu values differ  %s\n", d_differ,
                        d_differ ? "FAIL (dispatcher did not pick the VNNI kernel?)" : "ok");
            if (d_differ) ++failures;
        }
        for (int nt = 1; nt <= NT; ++nt) {
            const double us_a = time_us([&] { cpu::iq256_gu_rows(gt, blob.data(), gu_row, up_off, (int) H, ap, nt, ao, 0, (int) FF); }, 50);
            const double us_b = time_us([&] { cpu::iqvnni_gu_rows(gt, blob.data(), gu_row, up_off, (int) H, ap, nt, bo, 0, (int) FF); }, 50);
            std::printf("[21 gu ] %d token(s) one thread: iq256 %.0f us (%.2f GB/s), iqvnni %.0f us (%.2f GB/s), %.2fx\n",
                        nt, us_a, 2.0 * up_off / us_a / 1e3, us_b, 2.0 * up_off / us_b / 1e3, us_a / us_b);
        }
    }

    // ---- (2) IQ4_NL (20) down rows: iq4nl_vnni vs iq4nl_avx2 bitwise, vs ggml, times
    {
        const int dt = 20;
        std::vector<uint8_t> w = quant_rows(dt, H, FF, 20);
        const size_t d_row = ggml_row_size((ggml_type) dt, FF);
        cpu::NativeFmt f;
        std::string err;
        if (!cpu::native_fmt(21, dt, H, FF, f, err)) { std::printf("[20 dn ] native_fmt: %s\n", err.c_str()); return 1; }
        std::vector<float> h((size_t) NT * FF);
        for (auto& v : h) v = nd(rng);
        std::vector<std::vector<uint8_t>> hq(NT, std::vector<uint8_t>(cpu::kNativeHBytes));
        const void* hp[NT];
        for (int t = 0; t < NT; ++t) { cpu::native_quant_h(f, h.data() + (size_t) t * FF, hq[t].data()); hp[t] = hq[t].data(); }
        std::vector<std::vector<float>> a_out(NT, std::vector<float>(H)), b_out(NT, std::vector<float>(H));
        float* ao[NT]; float* bo[NT];
        for (int t = 0; t < NT; ++t) { ao[t] = a_out[t].data(); bo[t] = b_out[t].data(); }
        size_t differ = 0, values = 0;
        for (int nt = 1; nt <= NT; ++nt) {
            cpu::iq4nl256_down_rows(w.data(), d_row, (int) FF, hp, nt, ao, 0, (int) H);
            cpu::iq4nl256_down_rows_vnni(w.data(), d_row, (int) FF, hp, nt, bo, 0, (int) H);
            for (int t = 0; t < nt; ++t)
                for (int64_t r = 0; r < H; ++r) { ++values; differ += std::memcmp(&a_out[t][r], &b_out[t][r], 4) != 0; }
        }
        std::printf("[20 dn ] iq4nl vnni vs iq4nl avx2, 1..%d tokens: %zu of %zu values differ in any bit  %s\n", NT,
                    differ, values, differ ? "FAIL" : "ok");
        if (differ) ++failures;
        {   // against ggml-cpu on the same Q8_0 activations (measured 1e-7 in the AVX-2 case)
            const auto* tc = ggml_get_type_traits_cpu((ggml_type) dt);
            std::vector<float> got((size_t) NT * H), ref((size_t) NT * H);
            float* gp[NT];
            for (int t = 0; t < NT; ++t) gp[t] = got.data() + (size_t) t * H;
            cpu::iq4nl256_down_rows_vnni(w.data(), d_row, (int) FF, hp, NT, gp, 0, (int) H);
            for (int t = 0; t < NT; ++t)
                for (int64_t r = 0; r < H; ++r)
                    tc->vec_dot((int) FF, &ref[(size_t) t * H + r], 0, w.data() + (size_t) r * d_row, 0, hp[t], 0, 1);
            const double rd = rel(got, ref);
            std::printf("[20 dn ] iq4nl vnni down vs ggml: rel %.2e  %s\n", rd, rd > 1e-5 ? "FAIL" : "ok");
            if (rd > 1e-5) ++failures;
        }
        for (int nt = 1; nt <= NT; ++nt) {
            const double us_a = time_us([&] { cpu::iq4nl256_down_rows(w.data(), d_row, (int) FF, hp, nt, ao, 0, (int) H); }, 50);
            const double us_b = time_us([&] { cpu::iq4nl256_down_rows_vnni(w.data(), d_row, (int) FF, hp, nt, bo, 0, (int) H); }, 50);
            const double bytes = (double) d_row * H;
            std::printf("[20 dn ] %d token(s) one thread: avx2 %.0f us (%.2f GB/s), vnni %.0f us (%.2f GB/s), %.2fx\n",
                        nt, us_a, bytes / us_a / 1e3, us_b, bytes / us_b / 1e3, us_a / us_b);
        }
    }

    // ---- (3) Q2_0 (42) rows: vnni vs avx2 bitwise, times.  No ggml reference here: the GGUF rows kernel is
    // this project's own (ggml-cpu has no Q2_0 dot on x86).  Blocks are built by hand: d = 1.0, random codes.
    {
        const int64_t rows = H, cols = FF, nb = cols / 64;
        const size_t rb = (size_t) nb * 18;
        std::vector<uint8_t> w((size_t) rows * rb);
        for (int64_t r = 0; r < rows; ++r) {
            uint8_t* row = w.data() + (size_t) r * rb;
            for (int64_t b = 0; b < nb; ++b) {
                row[(size_t) b * 18] = 0x00;
                row[(size_t) b * 18 + 1] = 0x3C;   // fp16 1.0
                for (int k = 0; k < 16; ++k) row[(size_t) b * 18 + 2 + k] = (uint8_t) (rng() & 0xff);
            }
        }
        std::vector<float> x((size_t) NT * cols);
        for (auto& v : x) v = nd(rng);
        std::vector<cpu::ActQ> aq(NT);
        const cpu::ActQ* ap[NT];
        for (int t = 0; t < NT; ++t) { cpu::act_quant_any(x.data() + (size_t) t * cols, (int) cols, aq[t]); ap[t] = &aq[t]; }
        std::vector<std::vector<float>> a_out(NT, std::vector<float>(H)), b_out(NT, std::vector<float>(H));
        float* ao[NT]; float* bo[NT];
        for (int t = 0; t < NT; ++t) { ao[t] = a_out[t].data(); bo[t] = b_out[t].data(); }
        size_t differ = 0, values = 0;
        for (int nt = 1; nt <= NT; ++nt) {
            cpu::q2_0_gguf_rows_multi_avx2(w.data(), rb, (int) nb, ap, nt, ao, 0, (int) rows);
            cpu::q2_0_gguf_rows_multi_vnni(w.data(), rb, (int) nb, ap, nt, bo, 0, (int) rows);
            for (int t = 0; t < nt; ++t)
                for (int64_t r = 0; r < rows; ++r) { ++values; differ += std::memcmp(&a_out[t][r], &b_out[t][r], 4) != 0; }
        }
        std::printf("[42 q2 ] vnni vs avx2, 1..%d tokens: %zu of %zu values differ in any bit  %s\n", NT, differ, values,
                    differ ? "FAIL" : "ok");
        if (differ) ++failures;
        for (int nt = 1; nt <= NT; ++nt) {
            const double us_a = time_us([&] { cpu::q2_0_gguf_rows_multi_avx2(w.data(), rb, (int) nb, ap, nt, ao, 0, (int) rows); }, 50);
            const double us_b = time_us([&] { cpu::q2_0_gguf_rows_multi_vnni(w.data(), rb, (int) nb, ap, nt, bo, 0, (int) rows); }, 50);
            const double bytes = (double) rb * rows;
            std::printf("[42 q2 ] %d token(s) one thread: avx2 %.0f us (%.2f GB/s), vnni %.0f us (%.2f GB/s), %.2fx\n",
                        nt, us_a, bytes / us_a / 1e3, us_b, bytes / us_b / 1e3, us_a / us_b);
        }
    }

    std::printf("iqvnni_parity: %d failures\n", failures);
    return failures ? 1 : 0;
}
