// include/strata/kernels/cpu/iq_avxvnni.hpp - VEX-256 AVX-VNNI (`vpdpbusd`) dot products for the i-quant
// expert formats (the IQ3_S rows of format 21) and for IQ4_NL (20) against Q8_K / Q8_0 activations.
//
// The kernels of iq_avx2.cpp with `vpdpbusd` in place of maddubs+madd: the odd per-half scale sc = 2*s+1 is
// split exactly, sc*g*y = 2*(s*g)*y + (g*y), so two signed-VNNI passes replace the pair.  The integer sums
// are the same values in the same order, so every output is BITWISE equal to the AVX2 kernel's - checked
// by iqvnni_parity, and the reason this file cannot simply grow a third dpbusd variant for the IQ2 formats
// (their grid bytes go to 43, s to 15: the prescale would not fit a byte and would not be exact).
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

/// Formats with a VNNI kernel here: IQ3_S (21) for the gate/up and plain row dots - the only i-quant whose
/// grid bytes (<= 15) times the largest s (<= 15) stay in a byte - and IQ4_NL (20), for the down rows only.
bool iqvnni_supported(int ggml_type) noexcept;
/// ff[t][r] = silu(gate_r . a[t]) * (up_r . a[t]), rows [r0, r1); gate rows at blob, up rows at blob + up_off.
void iqvnni_gu_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                    int nt, float* const* ff, int r0, int r1);
/// out[t][r] = w_r . a[t], rows [r0, r1).
void iqvnni_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                 float* const* out, int r0, int r1);

/// out[t][r] = w_r . h[t] for IQ4_NL (type 20) rows against Q8_0 activations (ggml's block_q8_0): the
/// iq4nl256_down_rows kernel with `vpdpbusd` for maddubs+madd (bitwise equal).
void iq4nl256_down_rows_vnni(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt,
                             float* const* out, int r0, int r1);

}  // namespace strata::kernels::cpu
