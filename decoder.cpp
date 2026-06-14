/*
BGF decoder and new_variation bit-flip decoder, 
only support BIKE level 1
*/
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstdint>
#include <iostream>
#include <openssl/evp.h>
#include <openssl/sha.h>
#include "SHAKE256.h"
#include "key_generator.h"

// threshold function
int threshold(int S) {
    int result = static_cast<int>(0.0069722 * S + 13.530);
    return (result > 36) ? result : 36;
}

// old decoder.cpp function

// ctr function: calculates the upc value
int ctr(const std::vector<uint8_t>& h_i, const std::vector<uint8_t>& s, int j) {
    const int r = static_cast<int>(h_i.size());
    int c = 0;

    // directly extract the support of h_i internally without modifying the function input.
    for (int k = 0; k < r; ++k) {
        if (h_i[static_cast<size_t>(k)] != 0) {
            c += s[static_cast<size_t>((j + k) % r)];
        }
    }

    return c;
}

struct BFIterResult {
    std::vector<uint8_t> e;     // length 2r
    std::vector<uint8_t> black; // length 2r
    std::vector<uint8_t> gray;  // length 2r
};

// BFIter(bit-flip iteration) function implementation
// rely on ：ctr(h_i, s, j)
BFIterResult BFIter(const std::vector<uint8_t>& s,
                    const std::vector<uint8_t>& e_in,
                    int T, int tau,
                    const std::vector<std::vector<uint8_t>>& H) {
    const int r = static_cast<int>(s.size());

    BFIterResult out;
    out.e = e_in;
    out.black.assign(static_cast<size_t>(2 * r), 0);
    out.gray.assign(static_cast<size_t>(2 * r), 0);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < r; ++j) {
            int c = ctr(H[i], s, j);
            int idx = i * r + j;          //  mapped to [0..2r-1] of e

            if (c >= T) {
                out.e[idx] ^= 1;          // e_j <- e_j ⊕ 1
                out.black[idx] = 1;       // black_j <- 1
            } else if (c >= T - tau) {
                out.gray[idx] = 1;        // gray_j <- 1
            }
        }
    }

    return out;
}

// BFMaskedIter function
// "mask" is the "black" or "gray" value (with a length of 2r) returned by BFIter.
std::vector<uint8_t> BFMaskedIter(const std::vector<uint8_t>& s,
                                 const std::vector<uint8_t>& e_in,
                                 const std::vector<uint8_t>& mask,
                                 int T,
                                 const std::vector<std::vector<uint8_t>>& H) {
    const int r = static_cast<int>(s.size());

    std::vector<uint8_t> e = e_in;

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < r; ++j) {
            int c = ctr(H[i], s, j);
            int idx = i * r + j;

            if (c >= T) {
                e[idx] ^= mask[idx];   // e_j <- e_j ⊕ mask_j
            }
        }
    }

    return e;
}

// Calculate s = eH = e0*h0 + e1*h1  (in GF(2), cyclic convolution modulo x^r - 1; "+" represents bitwise XOR)
// length of e is 2r：e0 = e[0..r-1], e1 = e[r..2r-1]
// H is a two-dimensional array：H[0]=h0, H[1]=h1 （h_0, h_1 r）
std::vector<uint8_t> matrix_multicative(const std::vector<uint8_t>& e,
                                        const std::vector<std::vector<uint8_t>>& H) {
    const int r = static_cast<int>(H[0].size());

    std::vector<uint8_t> s(static_cast<size_t>(r), 0);

    // -----------------------------
    // Extract "support" within the function
    // -----------------------------
    std::vector<int> e0_support;
    std::vector<int> e1_support;
    std::vector<int> h0_support;
    std::vector<int> h1_support;

    e0_support.reserve(r);
    e1_support.reserve(r);
    h0_support.reserve(r);
    h1_support.reserve(r);

    for (int i = 0; i < r; ++i) {
        if (e[static_cast<size_t>(i)] != 0) {
            e0_support.push_back(i);
        }
        if (e[static_cast<size_t>(r + i)] != 0) {
            e1_support.push_back(i);
        }
        if (H[0][static_cast<size_t>(i)] != 0) {
            h0_support.push_back(i);
        }
        if (H[1][static_cast<size_t>(i)] != 0) {
            h1_support.push_back(i);
        }
    }

    // -----------------------------
    // s ^= e0 * h0
    // -----------------------------
    for (int ep : e0_support) {
        for (int hp : h0_support) {
            s[static_cast<size_t>((ep + hp) % r)] ^= 1;
        }
    }

    // -----------------------------
    // s ^= e1 * h1
    // -----------------------------
    for (int ep : e1_support) {
        for (int hp : h1_support) {
            s[static_cast<size_t>((ep + hp) % r)] ^= 1;
        }
    }

    return s;
}

//  ================================================================================
// BGF algorithm:
struct BGFDecodeResult {
    bool ok;
    std::vector<uint8_t> e;   // length 2r; If unsuccessful, it can be empty or still return the final e
};

// input：s = eH^t(length r), H=(h_0, h_1)(2*r), w(=h0/h1 weight，replace d)
BGFDecodeResult BGF(const std::vector<uint8_t>& s,
                        const std::vector<std::vector<uint8_t>>& H,
                        int w) {
    const int r = static_cast<int>(s.size());
    const int NbIter = 5;
    const int tau = 3;

    std::vector<uint8_t> e(static_cast<size_t>(2 * r), 0);

    for (int iter = 1; iter <= NbIter; ++iter) {
        // syn = s + eH^T  （GF(2)：bitwise XOR）
        std::vector<uint8_t> eH = matrix_multicative(e, H);
        std::vector<uint8_t> syn(static_cast<size_t>(r), 0);
        for (int k = 0; k < r; ++k) syn[static_cast<size_t>(k)] = s[static_cast<size_t>(k)] ^ eH[static_cast<size_t>(k)];

        // T = threshold(|syn|)
        int S = weight_u8(syn);
        int T = threshold(S);

        // (e, black, gray) = BFIter(syn, e, T, H)
        BFIterResult it = BFIter(syn, e, T, tau, H);
        e = it.e;

        // if iter == 1: e = BFMaskedIter(syn, e, black, (w+1)/2+1, H)
        if (iter == 1) {
            int Tmask = (w + 1) / 2 + 1;
            e = BFMaskedIter(syn, e, it.black, Tmask, H);
            e = BFMaskedIter(syn, e, it.gray,  Tmask, H);
        }
    }

    // verification: If s == eH^T, then it is successful.
    std::vector<uint8_t> final_eH = matrix_multicative(e, H);
    bool ok = true;
    for (int k = 0; k < r; ++k) {
        if (final_eH[static_cast<size_t>(k)] != s[static_cast<size_t>(k)]) { ok = false; break; }
    }

    return BGFDecodeResult{ok, e};
}

//  ================================================================================
// BIKE-Flip Algorithm (Latest Version in 2024):

// New threshold algorithm:
// level 1 only
// Inputs are all ints: S, i, S0, w (where d = w/2, δ = 3)
// threshold(S,i,S0,w) = max( f(S), T_i(S0) )
int bike_threshold(int S, int i, int S0, int w) {
    // Level 1 parameters
    const int delta = 3;
    const int d = w / 2;
    const double a = 0.006254868353074983;
    const double b = 11.101432337243956;

    auto f = [&](int x) -> double {       // f_134(x)
        return a * static_cast<double>(x) + b;
    };

    // caculate f(S)
    const int fS = static_cast<int>(f(S)); // Positive number truncation = floor

    // caculate T_i(S0)
    double Ti;
    if (i == 1) {
        Ti = f(S0) + delta;
    } else if (i == 2) {
        Ti = (1.0 / 3.0) * (2.0 * f(S0) + (static_cast<double>(d) + 1.0) / 2.0) + delta;
    } else if (i == 3) {
        Ti = (1.0 / 3.0) * (f(S0) + 2.0 * ((static_cast<double>(d) + 1.0) / 2.0)) + delta;
    } else { // i >= 4
        Ti = (static_cast<double>(d) + 1.0) / 2.0 + delta;
    }
    const int Ti_int = static_cast<int>(Ti); // Positive number truncation = floor

    return std::max(fS, Ti_int);
}


// Original BFIter:
// Input: s (length r), e (length 2r), H (2×r), T
// Output: updated e
std::vector<uint8_t> bike_BFIter(const std::vector<uint8_t>& s,
                                  const std::vector<uint8_t>& e_in,
                                  int T,
                                  const std::vector<std::vector<uint8_t>>& H) {
    const int r = static_cast<int>(s.size());
    std::vector<uint8_t> e = e_in;

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < r; ++j) {
            int c = ctr(H[i], s, j);
            int idx = i * r + j;
            if (c >= T) {
                e[idx] ^= 1;   // e_j <- e_j ⊕ 1
            }
        }
    }

    return e;
}

// BIKE-Flip algorithm:
struct BikeDecodeResult {
    bool ok;
    std::vector<uint8_t> e;   // length 2r
};

// Level 1：
// Input: s (length r), H (2×r), w (weights of h0/h1)
// Output: if successful, ok=true and return e; otherwise ok=false
BikeDecodeResult bike_decode(const std::vector<uint8_t>& s,
                                   const std::vector<std::vector<uint8_t>>& H,
                                   int w) {
    const int r = static_cast<int>(s.size());
    const int NbIter = 7;

    std::vector<uint8_t> e(static_cast<size_t>(2 * r), 0);

    for (int iter = 1; iter <= NbIter; ++iter) {
        // syn = s + eH^T  （GF(2)：bitwise XOR）
        std::vector<uint8_t> eH = matrix_multicative(e, H);
        std::vector<uint8_t> syn(static_cast<size_t>(r), 0);
        for (int k = 0; k < r; ++k) {
            syn[static_cast<size_t>(k)] = s[static_cast<size_t>(k)] ^ eH[static_cast<size_t>(k)];
        }

        // T = threshold(|syn|, iter, |s|)
        int T = bike_threshold(weight_u8(syn), iter, weight_u8(s), w);

        // e <- BFIter(syn, e, T, H)
        e = bike_BFIter(syn, e, T, H);
    }

    // if s == eH^T then return e else ⊥
    std::vector<uint8_t> final_eH = matrix_multicative(e, H);
    bool ok = true;
    for (int k = 0; k < r; ++k) {
        if (final_eH[static_cast<size_t>(k)] != s[static_cast<size_t>(k)]) {
            ok = false;
            break;
        }
    }

    return BikeDecodeResult{ok, e};
}

// ================================================================================
// The following are the optimized new functions

// =========================
// New: precomputed support of H
// =========================
struct DecoderPrecomp {
    int r = 0;
    std::vector<int> h_support[2];   // H[0], H[1] support
};

DecoderPrecomp build_decoder_precomp(const std::vector<std::vector<uint8_t>>& H) {
    DecoderPrecomp pc;
    pc.r = static_cast<int>(H[0].size());

    for (int block = 0; block < 2; ++block) {
        pc.h_support[block].reserve(pc.r);
        for (int i = 0; i < pc.r; ++i) {
            if (H[block][static_cast<size_t>(i)] != 0) {
                pc.h_support[block].push_back(i);
            }
        }
    }
    return pc;
}

// =========================
// // Thread-local / sample-reuse workspace
// =========================
struct DecoderScratch {
    std::vector<uint8_t> e;      // length 2r
    std::vector<uint8_t> eH;     // length r
    std::vector<uint8_t> s0;     // original syndrome，length r
    std::vector<uint8_t> syn;    // iteration syndrome，length r
    std::vector<uint8_t> black;  // length 2r
    std::vector<uint8_t> gray;   // length 2r

    std::vector<int> e0_support;
    std::vector<int> e1_support;

    explicit DecoderScratch(int r)
        : e(static_cast<size_t>(2 * r), 0),
          eH(static_cast<size_t>(r), 0),
          s0(static_cast<size_t>(r), 0),
          syn(static_cast<size_t>(r), 0),
          black(static_cast<size_t>(2 * r), 0),
          gray(static_cast<size_t>(2 * r), 0) {
        e0_support.reserve(r);
        e1_support.reserve(r);
    }
};

// =========================
// New: fast UPC calculation based on support
// =========================
int ctr_support(const std::vector<int>& support,
                       const uint8_t* s,
                       int j,
                       int r) {
    int c = 0;
    for (int hp : support) {
        int idx = j + hp;
        if (idx >= r) idx -= r;   // Because j, hp < r, so at most one subtraction
        c += s[idx];
    }
    return c;
}

// =========================
// New: extract support of e=(e0,e1)
// =========================
void extract_error_support(const std::vector<uint8_t>& e,
                                  int r,
                                  std::vector<int>& e0_support,
                                  std::vector<int>& e1_support) {
    e0_support.clear();
    e1_support.clear();

    for (int i = 0; i < r; ++i) {
        if (e[static_cast<size_t>(i)] != 0) {
            e0_support.push_back(i);
        }
        if (e[static_cast<size_t>(r + i)] != 0) {
            e1_support.push_back(i);
        }
    }
}

// =========================
// New: compute s = eH using support
// =========================
void matrix_multicative_fast(const std::vector<uint8_t>& e,
                                    const DecoderPrecomp& pc,
                                    std::vector<uint8_t>& s_out,
                                    std::vector<int>& e0_support,
                                    std::vector<int>& e1_support) {
    const int r = pc.r;
    std::fill(s_out.begin(), s_out.end(), 0);

    extract_error_support(e, r, e0_support, e1_support);

    for (int ep : e0_support) {
        for (int hp : pc.h_support[0]) {
            int idx = ep + hp;
            if (idx >= r) idx -= r;
            s_out[static_cast<size_t>(idx)] ^= 1;
        }
    }

    for (int ep : e1_support) {
        for (int hp : pc.h_support[1]) {
            int idx = ep + hp;
            if (idx >= r) idx -= r;
            s_out[static_cast<size_t>(idx)] ^= 1;
        }
    }
}

// =========================
// New: Here e_i has 1 only in the first half block, all zeros in the second half
// So s = e0 * h0, directly computed from ones_pos
// =========================
void matrix_multicative_first_half_from_positions(
    const std::vector<int>& ones_pos,
    const DecoderPrecomp& pc,
    std::vector<uint8_t>& s_out) {

    const int r = pc.r;
    std::fill(s_out.begin(), s_out.end(), 0);

    for (int ep : ones_pos) {
        for (int hp : pc.h_support[0]) {
            int idx = ep + hp;
            if (idx >= r) idx -= r;
            s_out[static_cast<size_t>(idx)] ^= 1;
        }
    }
}

// Compute s = eH = e0*h0 + e1*h1  (over GF(2), cyclic convolution modulo x^r - 1; "+" is bitwise XOR)
// e has length 2r: e0 = e[0..r-1], e1 = e[r..2r-1]
// H is a 2D array: H[0]=h0, H[1]=h1 (each of length r)

//  ================================================================================
// BGF algorithm:
// =========================
// New: only determine whether BGF succeeds, do not return e
// For use in batch statistics
// =========================
bool BGF_check_fast(const DecoderPrecomp& pc,
                           int w,
                           DecoderScratch& ws) {
    const int r = pc.r;
    const int NbIter = 5;
    const int tau = 3;

    std::fill(ws.e.begin(), ws.e.end(), 0);

    for (int iter = 1; iter <= NbIter; ++iter) {
        matrix_multicative_fast(ws.e, pc, ws.eH, ws.e0_support, ws.e1_support);

        for (int k = 0; k < r; ++k) {
            ws.syn[static_cast<size_t>(k)] =
                ws.s0[static_cast<size_t>(k)] ^ ws.eH[static_cast<size_t>(k)];
        }

        int T = threshold(weight_u8(ws.syn));

        if (iter == 1) {
            std::fill(ws.black.begin(), ws.black.end(), 0);
            std::fill(ws.gray.begin(), ws.gray.end(), 0);

            for (int block = 0; block < 2; ++block) {
                const auto& support = pc.h_support[block];
                for (int j = 0; j < r; ++j) {
                    int c = ctr_support(support, ws.syn.data(), j, r);
                    int idx = block * r + j;

                    if (c >= T) {
                        ws.e[static_cast<size_t>(idx)] ^= 1;
                        ws.black[static_cast<size_t>(idx)] = 1;
                    } else if (c >= T - tau) {
                        ws.gray[static_cast<size_t>(idx)] = 1;
                    }
                }
            }

            const int Tmask = (w + 1) / 2 + 1;

            for (int block = 0; block < 2; ++block) {
                const auto& support = pc.h_support[block];
                for (int j = 0; j < r; ++j) {
                    int c = ctr_support(support, ws.syn.data(), j, r);
                    int idx = block * r + j;
                    if (c >= Tmask) {
                        ws.e[static_cast<size_t>(idx)] ^= ws.black[static_cast<size_t>(idx)];
                    }
                }
            }

            for (int block = 0; block < 2; ++block) {
                const auto& support = pc.h_support[block];
                for (int j = 0; j < r; ++j) {
                    int c = ctr_support(support, ws.syn.data(), j, r);
                    int idx = block * r + j;
                    if (c >= Tmask) {
                        ws.e[static_cast<size_t>(idx)] ^= ws.gray[static_cast<size_t>(idx)];
                    }
                }
            }
        } else {
            for (int block = 0; block < 2; ++block) {
                const auto& support = pc.h_support[block];
                for (int j = 0; j < r; ++j) {
                    int c = ctr_support(support, ws.syn.data(), j, r);
                    if (c >= T) {
                        ws.e[static_cast<size_t>(block * r + j)] ^= 1;
                    }
                }
            }
        }
    }

    matrix_multicative_fast(ws.e, pc, ws.eH, ws.e0_support, ws.e1_support);

    for (int k = 0; k < r; ++k) {
        if (ws.eH[static_cast<size_t>(k)] != ws.s0[static_cast<size_t>(k)]) {
            return false;
        }
    }
    return true;
}

//  ================================================================================
// BIKE-Flip algorithm (2024 latest version):

// Original BFIter:
// Input: s (length r), e (length 2r), H (2×r), T
// Output: updated e
// =========================
// New: only determine whether BIKE-Flip succeeds, do not return e
// =========================
bool bike_decode_check_fast(const DecoderPrecomp& pc,
                                   int w,
                                   DecoderScratch& ws) {
    const int r = pc.r;
    const int NbIter = 7;
    const int S0 = weight_u8(ws.s0);

    std::fill(ws.e.begin(), ws.e.end(), 0);

    for (int iter = 1; iter <= NbIter; ++iter) {
        matrix_multicative_fast(ws.e, pc, ws.eH, ws.e0_support, ws.e1_support);

        for (int k = 0; k < r; ++k) {
            ws.syn[static_cast<size_t>(k)] =
                ws.s0[static_cast<size_t>(k)] ^ ws.eH[static_cast<size_t>(k)];
        }

        int T = bike_threshold(weight_u8(ws.syn), iter, S0, w);

        for (int block = 0; block < 2; ++block) {
            const auto& support = pc.h_support[block];
            for (int j = 0; j < r; ++j) {
                int c = ctr_support(support, ws.syn.data(), j, r);
                if (c >= T) {
                    ws.e[static_cast<size_t>(block * r + j)] ^= 1;
                }
            }
        }
    }

    matrix_multicative_fast(ws.e, pc, ws.eH, ws.e0_support, ws.e1_support);

    for (int k = 0; k < r; ++k) {
        if (ws.eH[static_cast<size_t>(k)] != ws.s0[static_cast<size_t>(k)]) {
            return false;
        }
    }
    return true;
}

