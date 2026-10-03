// src/kernels/cpu/iq_avxvnni.cpp - the i-quant expert rows on AVX-VNNI (VEX-256 `vpdpbusd`), several tokens
// at once.  See the header for why only IQ3_S (21) and the IQ4_NL (20) down rows live here.
//
// The AVX-2 kernel (iq_avx2.cpp) applies one 32-value half with a load, a `sign`, a `maddubs`, a `madd`
// (the scale sc = 2s+1) and an `add`.  `vpdpbusd` fuses the product and the i32 accumulation, but has no
// scale; the odd scale splits exactly:  sc*g*y = 2*(s*g)*y + (g*y).  Both operands stay in a byte when
// max grid byte * s <= 255 - true for iq3s_grid (max 15, verified over the table) with s <= 15: 225 <= 255,
// and the `mullo_epi16` prescale cannot carry across bytes (225*256 + 225 < 65536).  So one prescale per
// half (shared by all tokens of the window) and two `vpdpbusd` per token replace maddubs+madd+add, with
// the SAME integer values in the SAME 4-value lanes and the same float epilogue: iqvnni_* is bitwise equal
// to iq256_* (iqvnni_parity checks it, and #152's width invariance follows from it).
//
// The iq2*/iq3xxs grids go up to 43 / 62 in a byte, so s*byte would leave the byte and the split would not
// be exact: those formats keep the AVX-2 kernel (and an AVX-512 CPU runs them on iq512 already).
#include "strata/kernels/cpu/iq_avxvnni.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace strata::kernels::cpu {
namespace {

inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }
inline uint16_t u16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
inline uint64_t u64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

// 32 sign bits -> 32 bytes of -1 (bit set) / +1: ggml's bit_selector pattern, shared by all tokens.  A copy
// of iq_avx2.cpp's helper on purpose: this file IS the ISA boundary, and the two must stay separable.
inline __m256i sgn_vec(uint32_t m) {
    const __m128i bm = _mm_set1_epi32((int) m);
    const __m128i lo = _mm_shuffle_epi8(bm, _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1));
    const __m128i hi = _mm_shuffle_epi8(bm, _mm_setr_epi8(2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3));
    const __m256i bits = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
    const __m256i sel = _mm256_setr_epi8(1, 2, 4, 8, 16, 32, 64, (char) 0x80, 1, 2, 4, 8, 16, 32, 64, (char) 0x80,
                                         1, 2, 4, 8, 16, 32, 64, (char) 0x80, 1, 2, 4, 8, 16, 32, 64, (char) 0x80);
    const __m256i nz = _mm256_cmpeq_epi8(_mm256_and_si256(bits, sel), sel);
    return _mm256_or_si256(nz, _mm256_set1_epi8(1));
}

// keven_signs_q2xs (ggml keeps it static in arch/x86/quants.c): one u64 per 7-bit sign index, byte k
// = 0xFF when bit k of ksigns_iq2xs[i] is set, 0x01 otherwise.  Built once at load time from the shared
// ggml-common table so it can never drift from it.  (The same helper as iq_avx2.cpp's, see above.)
struct EvenSigns {
    uint64_t v[128];
    EvenSigns() {
        for (int i = 0; i < 128; ++i) {
            uint64_t r = 0;
            for (int k = 0; k < 8; ++k)
                r |= (uint64_t) (((ksigns_iq2xs[i] >> k) & 1) ? 0xFF : 0x01) << (8 * k);
            v[i] = r;
        }
    }
};
static const EvenSigns even_signs;

inline float hsum8(__m256 v) {
    const __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

// E-2 (iq_avx2.cpp): the pool streams the expert rows from DRAM, so ask for the bytes ahead of the decode.
// Same switch: STRATA_IQ_PREFETCH is the distance in bytes, 0 = off, default 2048.
const int prefetch_ahead = [] {
    const char* v = std::getenv("STRATA_IQ_PREFETCH");
    return v ? std::atoi(v) : 2048;
}();

inline void rows_ahead(const uint8_t* p) {
    if (prefetch_ahead <= 0) return;
    _mm_prefetch((const char*) p, _MM_HINT_T0);
    _mm_prefetch((const char*) p + 64, _MM_HINT_T0);
}

// ---- IQ3_S (21): one 32-value half -> g (the grid bytes), gs = g*s (the "2s" VNNI operand) and sgn.
// sc = 2s+1 is never built: the "+1" pass is g itself, the "2s" pass the prescaled vector.
inline void decode21(const uint8_t* b, int j, int half, __m256i& g, __m256i& gs, __m256i& sgn) {
    const uint8_t* q = b + 2 + 16 * j + 8 * half;
    const uint32_t h = b[66 + 2 * j + half];
#define G3(k) (int) iq3s_grid[q[k] | (((h >> k) & 1u) << 8)]
    g = _mm256_set_epi32(G3(7), G3(6), G3(5), G3(4), G3(3), G3(2), G3(1), G3(0));
#undef G3
    const uint64_t m = u64(b + 74 + 8 * j);
    sgn = sgn_vec(half ? (uint32_t) (m >> 32) : (uint32_t) m);
    const uint8_t s = b[106 + j];
    // the prescale: iq3s_grid bytes <= 15, s <= 15, so 225 <= 255 - a byte, and the 16-bit lanes of
    // mullo_epi16 cannot carry a multiply into their neighbor byte (225*256+225 < 65536)
    gs = _mm256_mullo_epi16(g, _mm256_set1_epi16((short) (half ? (s >> 4) : (s & 15))));
}

template <int NT>
inline void row_dot21(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
    const __m256i zero = _mm256_setzero_si256();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * 110;   // sizeof(block_iq3_s): QK_K = 256 weights
        rows_ahead(blk + prefetch_ahead);
        // the block's two VNNI accumulators: a = sum(s*g*y), b = sum(g*y); 2a+b is the block's integer sum
        // with the same 4-value lanes the AVX-2 kernel's madd produces
        __m256i acc_a[NT], acc_b[NT];
        for (int t = 0; t < NT; ++t) acc_a[t] = acc_b[t] = zero;
        for (int j = 0; j < 4; ++j) {
            for (int half = 0; half < 2; ++half) {
                __m256i g, gs, sgn;
                decode21(blk, j, half, g, gs, sgn);
                const int off = 64 * j + 32 * half;
                for (int t = 0; t < NT; ++t) {
                    const __m256i yv = _mm256_loadu_si256((const __m256i*) (y[t][i].qs + off));
                    const __m256i ys = _mm256_sign_epi8(yv, sgn);
                    acc_a[t] = _mm256_dpbusd_avx_epi32(acc_a[t], gs, ys);   // the 2s*y part
                    acc_b[t] = _mm256_dpbusd_avx_epi32(acc_b[t], g, ys);    // the 1*y part
                }
            }
        }
        const float dx = h2f(u16(blk));   // Fmt32<21>::K = 1
        for (int t = 0; t < NT; ++t) {
            const __m256i tot = _mm256_add_epi32(_mm256_add_epi32(acc_a[t], acc_a[t]), acc_b[t]);
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d), _mm256_cvtepi32_ps(tot), accf[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum8(accf[t]);
}

template <int NT>
void gu_rows21(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, float* const* ff,
               int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    const int nb = n / QK_K;
    float g[NT], u[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot21<NT>(blob + (size_t) r * gu_row, nb, y, g);
        row_dot21<NT>(blob + up_off + (size_t) r * gu_row, nb, y, u);
        for (int t = 0; t < NT; ++t) ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
    }
}

template <int NT>
void dot_rows21(const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot21<NT>(w + (size_t) r * row_bytes, n / QK_K, y, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

void gu_rows21_nt(int nt, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                  float* const* ff, int r0, int r1) {
    switch (nt) {
        case 1: gu_rows21<1>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 2: gu_rows21<2>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 3: gu_rows21<3>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 4: gu_rows21<4>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 5: gu_rows21<5>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 6: gu_rows21<6>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 7: gu_rows21<7>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        default: gu_rows21<8>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
    }
}

void dot_rows21_nt(int nt, const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out,
                   int r0, int r1) {
    switch (nt) {
        case 1: dot_rows21<1>(w, row_bytes, n, act, out, r0, r1); break;
        case 2: dot_rows21<2>(w, row_bytes, n, act, out, r0, r1); break;
        case 3: dot_rows21<3>(w, row_bytes, n, act, out, r0, r1); break;
        case 4: dot_rows21<4>(w, row_bytes, n, act, out, r0, r1); break;
        default: for (int t0 = 0; t0 < nt; t0 += 4) {
            const int k = nt - t0 < 4 ? nt - t0 : 4;
            dot_rows21_nt(k, w, row_bytes, n, act + t0, out + t0, r0, r1);
        }
    }
}

// ---- IQ4_NL (20) down rows against Q8_0 activations: iq4nl_rows (iq_avx2.cpp) with `vpdpbusd` for
// maddubs+madd(ones).  aq = |w| (<= 8, a byte) and ys = the signed activation: one VNNI dot IS the sum -
// the scale is the float epilogue's, exactly as in the AVX-2 kernel.  Bitwise equal to it.
template <int NT>
void iq4nl_rows_vnni(const uint8_t* w, size_t row_bytes, int n, const block_q8_0* const* y, float* const* out,
                     int r0, int r1) {
    const __m128i values = _mm_loadu_si128((const __m128i*) kvalues_iq4nl);
    const __m128i m4b = _mm_set1_epi8(0x0f);
    const __m256i zero = _mm256_setzero_si256();
    const int nb = n / QK4_NL;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * row_bytes;
        __m256 accf[NT];
        for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
        for (int ib = 0; ib < nb; ++ib) {
            const uint8_t* blk = row + (size_t) ib * sizeof(block_iq4_nl);
            rows_ahead(blk + prefetch_ahead);
            const __m128i bits = _mm_loadu_si128((const __m128i*) (blk + 2));
            const __m128i lo = _mm_and_si128(bits, m4b);                      // values 0..15
            const __m128i hi = _mm_and_si128(_mm_srli_epi16(bits, 4), m4b);    // values 16..31
            const __m256i q4 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_shuffle_epi8(values, lo)),
                                                       _mm_shuffle_epi8(values, hi), 1);
            const __m256i aq = _mm256_sign_epi8(q4, q4);   // |w|: the unsigned operand of the VNNI dot
            const float dx = h2f(u16(blk));
            for (int t = 0; t < NT; ++t) {
                const block_q8_0& b = y[t][ib];
                const __m256i q8 = _mm256_loadu_si256((const __m256i*) b.qs);
                const __m256i p = _mm256_dpbusd_avx_epi32(zero, aq, _mm256_sign_epi8(q8, q4));
                accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * h2f(b.d)), _mm256_cvtepi32_ps(p), accf[t]);
            }
        }
        for (int t = 0; t < NT; ++t) out[t][r] = hsum8(accf[t]);
    }
}

}  // namespace

bool iqvnni_supported(int type) noexcept { return type == 21; }

void iqvnni_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                    int nt, float* const* ff, int r0, int r1) {
    switch (type) {
        case 21: gu_rows21_nt(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        default: break;
    }
}

void iqvnni_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                 int r0, int r1) {
    switch (type) {
        case 21: dot_rows21_nt(nt, w, row_bytes, n, act, out, r0, r1); break;
        default: break;
    }
}

void iq4nl256_down_rows_vnni(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt, float* const* out,
                             int r0, int r1) {
    const block_q8_0* y[8];
    for (int t = 0; t < nt; ++t) y[t] = (const block_q8_0*) hq[t];
    switch (nt) {
        case 1: iq4nl_rows_vnni<1>(w, row_bytes, n, y, out, r0, r1); break;
        case 2: iq4nl_rows_vnni<2>(w, row_bytes, n, y, out, r0, r1); break;
        case 3: iq4nl_rows_vnni<3>(w, row_bytes, n, y, out, r0, r1); break;
        case 4: iq4nl_rows_vnni<4>(w, row_bytes, n, y, out, r0, r1); break;
        default: for (int t0 = 0; t0 < nt; t0 += 4) {
            const int k = nt - t0 < 4 ? nt - t0 : 4;
            iq4nl256_down_rows_vnni(w, row_bytes, n, hq + t0, k, out + t0, r0, r1);
        }
    }
}

}  // namespace strata::kernels::cpu
