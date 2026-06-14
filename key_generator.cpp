/* 
We have implemented not only the random keys, Type 1 weak keys and (m,epsilon)-cluster weak keys 
adopted in the paper, but also Type 2 and Type 3 weak keys, as well as keys satisfying the structure 
wt(h_0 && h_1^{inv}) = m.
*/
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm> // std::find
#include <stdexcept>
#include "SHAKE256.h" //
#include <cstdint>
#include <numeric>   // std::gcd

// 128-bit security level : r=12323 w=142/2=71 t=134 n=2r
// 80-bit security level : r=4801 w=90 t=84

// Generate wlist (of size w), with element range in [0, r-1], and also applicable to the list of incorrect positions
std::vector<int> generate_wlist(int r, int w, const std::string& seed) {

    std::vector<int> wlist;
    wlist.reserve(static_cast<size_t>(w)); 
    //reserve(n)：pre-allocate memory for the vector to accommodate n elements
    // static_cast<size_t>(w)：Convert w to an unsigned integer type

    EVP_MD_CTX* ctx = prng_init_shake256(seed);
    if (!ctx) throw std::runtime_error("prng_init_shake256 returned nullptr"); // Check whether the PRNG context is valid

    // for (i=w-1; i>=0; i--)
    for (int i = w - 1; i >= 0; --i) {
        int bound = r - i;                 // r - i > 0，because i <= w-1 and w <= r
        int offset = randint_uniform(bound, ctx); // in [0, bound-1]
        int pos = i + offset;              // in [i, r-1]

        // If pos is in wlist, then add i to wlist; otherwise, add pos to wlist.
        if (std::find(wlist.begin(), wlist.end(), pos) != wlist.end()) {
            wlist.push_back(i);
        } else {
            wlist.push_back(pos);
        }
    }
    EVP_MD_CTX_free(ctx);
    return wlist;
}

//-------------------------------------------------------------------------------------------------------------
// Generate a random polynomial h_0 (or h_1) with a length of r and a weight of w/2.
std::vector<uint8_t> generate_random_poly(int r, int w, const std::string& seed) {
    // Step 1: Call the function "generate_wlist" to obtain the position of 1
    std::vector<int> wlist = generate_wlist(r, w, seed);

    // Step 2: Initialize a length-r string consisting entirely of zeros
    std::vector<uint8_t> poly(static_cast<size_t>(r), 0);

    // Step 3: Fill in 1 at the designated position
    for (int pos : wlist) {
        poly[pos] = 1;
    }
    // pos : The wlist iterates through each element in the wlist container. During each iteration, 
    // the value of the current element is assigned to the variable pos, 
    // starting from the first element and ending at the last element.
    return poly;
}

//-------------------------------------------------------------------------------------------------------------
// Generate weak key type-1
std::vector<uint8_t> generate_weak1_poly(int r, int w, int f, const std::string& seed) {
    // int f : Fix the first f positions as 1 (0..f-1)
    // Initialize h
    std::vector<uint8_t> h(static_cast<size_t>(r), 0);

    // Fix the first f positions as 1: h[0..f-1] = 1
    for (int i = 0; i < f; ++i) {
        h[static_cast<size_t>(i)] = 1;
    }

    if (w - f > 0) {
        // Map back to the global level：pos_global = f + pos_tail
        std::vector<int> tail_pos = generate_wlist(r - f, w - f, seed);

        for (int p : tail_pos) {
            h[static_cast<size_t>(f + p)] = 1;
        }
    }

    return h;
}

//-------------------------------------------------------------------------------------------------------------
// Generate weak key type-2: r is the block size, w is the weight, 
// and m is the multiplicity of the distance spectrum. 
// u(delta, h)
std::vector<uint8_t> generate_weak2_poly(int r, int w, int m, const std::string& seed_a, const std::string& seed_b)
{
    int s = w - m;

    // Step 1: Generate sequence a
    // Select s-1 positions from the range [0, d-2], and then replace +1 with [1, d-1]
    std::vector<int> a_mid;
    if (s - 1 > 0) {
        a_mid = generate_wlist(w - 1, s - 1, seed_a);
        for (auto& x : a_mid)
            x += 1;  // shift to [1, w-1]
        std::sort(a_mid.begin(), a_mid.end());
    }

    std::vector<int> a(s + 1);
    a[0] = 0;
    for (int i = 1; i <= s - 1; ++i)
        a[i] = a_mid[i - 1];
    a[s] = w;

    // Step 2: Generate sequence b

    std::vector<int> b_mid;
    if (s - 1 > 0) {
        b_mid = generate_wlist(r - w - 1, s - 1, seed_b);
        for (auto& x : b_mid)
            x += 1;  // shift to [1, r-w-1]
        std::sort(b_mid.begin(), b_mid.end());
    }

    std::vector<int> b(s + 1);
    b[0] = 0;
    for (int i = 1; i <= s - 1; ++i)
        b[i] = b_mid[i - 1];
    b[s] = r - w;

    // Step 3: Calculate the difference

    std::vector<int> o(s + 1);
    std::vector<int> z(s + 1);

    for (int j = 1; j <= s; ++j) {
        o[j] = a[j] - a[j - 1];
        z[j] = b[j] - b[j - 1];
    }

    // Step 4: Construct h (loop index)

    std::vector<uint8_t> h(static_cast<size_t>(r), 0);

    int i = 0;

    for (int j = 1; j <= s; ++j) {

        i += z[j];

        for (int k = 0; k < o[j]; ++k) {
            int idx = (i + k) % r;   // loop index
            h[static_cast<size_t>(idx)] = 1;
        }

        i += o[j];
    }

    return h;
}

//-------------------------------------------------------------------------------------------------------------
// Generate weak key type-3
// Right shift (cyclic right shift) bit vector: new[(i+l)%r] = old[i]
std::vector<uint8_t> rotate_right(const std::vector<uint8_t>& v, int l) {
    const int r = static_cast<int>(v.size());
    l %= r;
    if (l < 0) l += r;

    std::vector<uint8_t> out(static_cast<size_t>(r), 0);
    for (int i = 0; i < r; ++i) {
        int j = (i + l) % r;
        out[static_cast<size_t>(j)] = v[static_cast<size_t>(i)];
    }
    return out;
}

// // Randomly select m elements from "items" without replacement (using SHAKE PRNG)
static std::vector<int> sample_m_from_list(const std::vector<int>& items, int m, const std::string& seed) {
    std::vector<int> idx(items.size());
    for (size_t i = 0; i < items.size(); ++i) idx[i] = static_cast<int>(i);

    EVP_MD_CTX* ctx = prng_init_shake256(seed);
    if (!ctx) throw std::runtime_error("prng_init_shake256 returned nullptr");

    // Fisher-Yates shuffling is performed m times before the process begins.
    for (int i = 0; i < m; ++i) {
        int bound = static_cast<int>(idx.size()) - i;      // >0
        int off = randint_uniform(bound, ctx);             // [0, bound-1]
        int j = i + off;                                   // [i, size-1]
        std::swap(idx[static_cast<size_t>(i)], idx[static_cast<size_t>(j)]);
    }
    EVP_MD_CTX_free(ctx);

    std::vector<int> picked;
    picked.reserve(static_cast<size_t>(m));
    for (int i = 0; i < m; ++i) {
        picked.push_back(items[static_cast<size_t>(idx[static_cast<size_t>(i)])]);
    }
    return picked;
}

// Write the position set pos into a 0/1 vector
static std::vector<uint8_t> positions_to_bits(int r, const std::vector<int>& pos) {
    std::vector<uint8_t> h(static_cast<size_t>(r), 0);
    for (int p : pos) {
        if (p < 0 || p >= r) throw std::runtime_error("positions_to_bits: position out of range");
        h[static_cast<size_t>(p)] = 1;
    }
    return h;
}

// Generate weak3:
// 1) Generate h0 (weight d)
// 2) Select m positions from the 1-positions of h0, copy to the same positions in h1
// 3) Fill the remaining 1s in h1 (total weight reaches d, without conflicting with copied positions)
// 4) Cyclic right shift h1 by l
//
// Returns (h0_bits, h1_bits_shifted)
std::pair<std::vector<uint8_t>, std::vector<uint8_t>> generate_weak3_poly(int r, int d, int m, const std::string& seed_base) {
    // ---- Step 0: Select shift l \in {0,...,r-1}
    EVP_MD_CTX* ctx_l = prng_init_shake256(seed_base + "|l");
    if (!ctx_l) throw std::runtime_error("prng_init_shake256 returned nullptr");
    int l = randint_uniform(r, ctx_l);
    EVP_MD_CTX_free(ctx_l);

    // ---- Step 1: Generate the 1's position (size d) of h0
    std::vector<int> h0_pos = generate_wlist(r, d, seed_base + "|h0");
    // Remove duplicates (theoretically, generate_wlist should not have any repetitions)
    std::sort(h0_pos.begin(), h0_pos.end());
    h0_pos.erase(std::unique(h0_pos.begin(), h0_pos.end()), h0_pos.end());
    if (static_cast<int>(h0_pos.size()) != d) {
        throw std::runtime_error("generate_wlist produced duplicates for h0 (unexpected)");
    }

    // ---- Step 2: Randomly select m items from h0_pos as the shared positions
    std::vector<int> shared = sample_m_from_list(h0_pos, m, seed_base + "|share");

    // Mark the occupied position of h1 with a Boolean table
    std::vector<uint8_t> used(static_cast<size_t>(r), 0);
    for (int p : shared) used[static_cast<size_t>(p)] = 1;

    // ---- Step 3: Generate h1 and the remaining (d-m) positions of 1, while avoiding shared.
    std::vector<int> h1_pos = shared;
    h1_pos.reserve(static_cast<size_t>(d));

    // Here, we use the "try-retry" approach: repeatedly generate the wlist using different seeds, 
    // and select the positions that do not conflict with each other.
    int need = d - m;
    int round = 0;
    while (need > 0) {
        std::string seed_extra = seed_base + "|h1extra|" + std::to_string(round++);
        std::vector<int> cand = generate_wlist(r, need, seed_extra);

        for (int p : cand) {
            if (p < 0 || p >= r) throw std::runtime_error("h1 candidate out of range");
            if (!used[static_cast<size_t>(p)]) {
                used[static_cast<size_t>(p)] = 1;
                h1_pos.push_back(p);
                --need;
                if (need == 0) break;
            }
        }
        // If all the candidates in this round fail, then proceed to the next round of seed_extra.
    }

    // h0/h1 Convert to bit vector
    std::vector<uint8_t> h0_bits = positions_to_bits(r, h0_pos);
    std::vector<uint8_t> h1_bits = positions_to_bits(r, h1_pos);

    // ---- Step 4: h1 Circular right shift l
    std::vector<uint8_t> h1_shifted = rotate_right(h1_bits, l);

    return {h0_bits, h1_shifted};
}


// m-cluster property weak key epsilon = 0 and 1
// epsilon=0

// ===== Generate epsilon=0 m-cluster polynomial h0: all 1s in [0, m-1], rest 0 =====
// r: polynomial length (usually r in BIKE)
// w: number of 1s (weight)
// m: cluster window length (window is [0, m-1])
// seed: seed for generating random positions
std::vector<uint8_t> generate_m_cluster_poly_0(int r, int w, int m, const std::string& seed) {

    // all 0
    std::vector<uint8_t> h0(static_cast<size_t>(r), 0);

    // Select w ones only within a window of length m
    std::vector<int> pos = generate_wlist(m, w, seed);

    for (int p : pos) {
        if (p < 0 || p >= m) {
            throw std::runtime_error("generate_wlist returned out-of-range position");
        }
        h0[static_cast<size_t>(p)] = 1;
    }
    return h0;
}

// epsilon=1
// Place w-1 ones within the range [0, m-1]; then add one more 1 within the range [m, r-1]
std::vector<uint8_t> generate_m_cluster_poly_1(int r, int w, int m, const std::string& seed) {

    std::vector<uint8_t> h0(static_cast<size_t>(r), 0);

    // 1) Place w - 1 ones in the first m positions
    if (w - 1 > 0) {
        std::vector<int> pos_in = generate_wlist(m, w - 1, seed + "/in");
        for (int p : pos_in) {
            if (p < 0 || p >= m) throw std::runtime_error("generate_wlist(in) out of range");
            h0[static_cast<size_t>(p)] = 1;
        }
    }

    // 2) Place 1 one in the last r - m positions (mapping the positions to the global range [m, r - 1])
    {
        std::vector<int> pos_out = generate_wlist(r - m, 1, seed + "/out");
        int p = pos_out[0]; // 0..(r-m-1)
        if (p < 0 || p >= (r - m)) throw std::runtime_error("generate_wlist(out) out of range");
        h0[static_cast<size_t>(m + p)] = 1;
    }

    return h0;
}


//-------------------------------------------------------------------------------------------------------------
// Private key similar to weak3 (other correlation properties of the two sequences, e.g., reversal correlation, etc.)
// Reversal correlation
// ====== Utility: left-right reversal of a bit vector ======
// out[i] = v[r-1-i]
static std::vector<uint8_t> flip_lr(const std::vector<uint8_t>& v) {
    const int r = static_cast<int>(v.size());
    std::vector<uint8_t> out(static_cast<size_t>(r), 0);
    for (int i = 0; i < r; ++i) {
        out[static_cast<size_t>(i)] = v[static_cast<size_t>(r - 1 - i)];
    }
    return out;
}

// ====== Generate key: Output (h0, h1_inv) such that wt(h0 && h1_inv) = m ======
std::pair<std::vector<uint8_t>, std::vector<uint8_t>> generate_flip_intersection_key(int r, int d, int m, const std::string& seed_base) {

    // ---- Step 1: Generate the 1's position (size d) of h0
    std::vector<int> h0_pos = generate_wlist(r, d, seed_base + "|h0");
    std::sort(h0_pos.begin(), h0_pos.end());
    h0_pos.erase(std::unique(h0_pos.begin(), h0_pos.end()), h0_pos.end());
    if ((int)h0_pos.size() != d) throw std::runtime_error("h0 duplicates from generate_wlist");

    // ---- Step 2: Select m elements from h0_pos to form the set S 
    // (in the coordinate system of h0) as the "shared positions after flipping"
    std::vector<int> S = sample_m_from_list(h0_pos, m, seed_base + "|S");

    // ---- Step 3: Construct the unflipped h1: First, insert flip(S) into it
    // Let p' = r - 1 - p. This ensures that flip(h1)[p] = h1[p'] = 1, thereby making (h0 && flip(h1)) overlap at p.
    std::vector<uint8_t> used((size_t)r, 0);
    std::vector<int> h1_pos;
    h1_pos.reserve((size_t)d);

    for (int p : S) {
        int pprime = (r - 1 - p);
        if (pprime < 0 || pprime >= r) throw std::runtime_error("pprime out of range");
        if (!used[(size_t)pprime]) {
            used[(size_t)pprime] = 1;
            h1_pos.push_back(pprime);
        } else {
            throw std::runtime_error("unexpected collision in flip(S)");
        }
    }
    if ((int)h1_pos.size() != m) throw std::runtime_error("h1_pos size mismatch after placing flip(S)");

    // ---- Step 4: Fill in the remaining (d-m) 1's in the h1 section, avoiding the already used positions.
    int need = d - m;
    int round = 0;
    while (need > 0) {
        std::string seed_extra = seed_base + "|h1extra|" + std::to_string(round++);
        std::vector<int> cand = generate_wlist(r, need, seed_extra);
        for (int p : cand) {
            if (p < 0 || p >= r) throw std::runtime_error("h1 extra candidate out of range");
            if (!used[(size_t)p]) {
                used[(size_t)p] = 1;
                h1_pos.push_back(p);
                --need;
                if (need == 0) break;
            }
        }
    }

    // ---- Step 5: Rotate the bit and then flip it to obtain h1_inv
    std::vector<uint8_t> h0_bits = positions_to_bits(r, h0_pos);
    std::vector<uint8_t> h1_bits = positions_to_bits(r, h1_pos);
    std::vector<uint8_t> h1_inv_bits = flip_lr(h1_bits);

    return {h0_bits, h1_inv_bits}; 
}

//-------------------------------------------------------------------------------------------------------------
// Weak keys that may be mentioned in other articles. As well as other types of sequences 
// (which lack randomness and have some structure) etc. (Briefly)(Currently known examples include the symmetric weak keys)


//-------------------------------------------------------------------------------------------------------------
int weight_u8(const std::vector<uint8_t>& e) {
    int w = 0;
    for (uint8_t x : e) w += (x != 0);
    return w;
}

// =============================================================================
// Filter out errors with similar m-aggregation properties and weak keys from random errors

// Generate a completely random error vector e with a length of 2r and a total weight of t
std::vector<uint8_t> generate_random_error_2r(int r, int t, const std::string& seed) {
    // Randomly select t positions on the interval [0, 2r-1]
    std::vector<int> wlist = generate_wlist(2 * r, t, seed);

    // A vector of length 2r consisting entirely of zeros
    std::vector<uint8_t> e(static_cast<size_t>(2 * r), 0);

    // Set the selected position to 1
    for (int pos : wlist) {
        e[static_cast<size_t>(pos)] = 1;
    }

    return e;
}

// Generate a random integer i such that the greatest common divisor of i and r is 1
int generate_random_coprime_i(int r, const std::string& seed) {
    // First, generate a candidate set, and then select the first number from it that is coprime with r
    // Simply perform linear probing based on the seed
    std::vector<int> candidates = generate_wlist(r - 1, 32, seed + "|phi_i_candidates");

    for (int x : candidates) {
        int i = x + 1;  // 映射到 [1, r-1]
        if (std::gcd(i, r) == 1) {
            return i;
        }
    }

    // If no candidates are found in the list, then scan through them in sequence again to 
    // ensure that a candidate will definitely be found.
    for (int i = 1; i < r; ++i) {
        if (std::gcd(i, r) == 1) {
            return i;
        }
    }

    throw std::runtime_error("failed to find i coprime with r");
}

// Apply to the polynomial Phi_i: f(x) -> f(x^i)
std::vector<uint8_t> apply_phi_i(const std::vector<uint8_t>& f, int r, int i) {
    if (static_cast<int>(f.size()) != r) {
        throw std::runtime_error("apply_phi_i: input size mismatch");
    }
    if (i <= 0 || i >= r || std::gcd(i, r) != 1) {
        throw std::runtime_error("apply_phi_i: i must satisfy 1 <= i < r and gcd(i, r) = 1");
    }

    std::vector<uint8_t> mapped(static_cast<size_t>(r), 0);

    for (int p = 0; p < r; ++p) {
        if (f[static_cast<size_t>(p)] != 0) {
            int new_pos = static_cast<int>((1LL * p * i) % r);
            mapped[static_cast<size_t>(new_pos)] = 1;
        }
    }

    return mapped;
}

// After generating the (m, epsilon=0)-cluster polynomial, then apply the random Phi_i
std::vector<uint8_t> generate_m_cluster_poly_0_phi(int r, int w, int m, const std::string& seed) {

    // First generate the original (m, epsilon=0)-cluster polynomial
    std::vector<uint8_t> h0(static_cast<size_t>(r), 0);

    // Select w ones only within a window of length m
    std::vector<int> pos = generate_wlist(m, w, seed + "|cluster");

    for (int p : pos) {
        if (p < 0 || p >= m) {
            throw std::runtime_error("generate_wlist returned out-of-range position");
        }
        h0[static_cast<size_t>(p)] = 1;
    }

    // Randomly select an i that is coprime to r
    int i = generate_random_coprime_i(r, seed + "|phi_i");

    // Phi_i: f(x) -> f(x^i)
    return apply_phi_i(h0, r, i);
}

std::vector<uint8_t> generate_m_cluster_poly_1_phi(int r, int w, int m, const std::string& seed) {

    std::vector<uint8_t> h0(static_cast<size_t>(r), 0);

    // 1) Place w - 1 ones in the first m positions
    if (w - 1 > 0) {
        std::vector<int> pos_in = generate_wlist(m, w - 1, seed + "/in");
        for (int p : pos_in) {
            if (p < 0 || p >= m) throw std::runtime_error("generate_wlist(in) out of range");
            h0[static_cast<size_t>(p)] = 1;
        }
    }

    // 2) Place 1 one at the last r - m positions (mapping the positions to the global range [m, r - 1])
    {
        std::vector<int> pos_out = generate_wlist(r - m, 1, seed + "/out");
        int p = pos_out[0]; // 0..(r-m-1)
        if (p < 0 || p >= (r - m)) throw std::runtime_error("generate_wlist(out) out of range");
        h0[static_cast<size_t>(m + p)] = 1;
    }

    // Randomly select an i that is coprime to r
    int i = generate_random_coprime_i(r, seed + "|phi_i");

    // 施加 Phi_i: f(x) -> f(x^i)
    return apply_phi_i(h0, r, i);
}

// --------------------------------------------------
// Determine whether it is a (m, epsilon=0)-cluster (in a looping sense)
// That is whether there exists a loop window of length m that contains all w ones
// --------------------------------------------------
bool is_m_cluster_poly_0_cyclic(const std::vector<uint8_t>& poly, int w, int m) {
    const int r = static_cast<int>(poly.size());

    if (r <= 0) {
        throw std::runtime_error("is_m_cluster_poly_0_cyclic: r must be positive");
    }
    if (m <= 0 || m > r) {
        throw std::runtime_error("is_m_cluster_poly_0_cyclic: m must satisfy 1 <= m <= r");
    }

    // Check if the total weight is w
    int total = 0;
    for (int i = 0; i < r; ++i) {
        total += (poly[static_cast<size_t>(i)] != 0);
    }
    if (total != w) {
        return false;
    }

    // The first window [0, m-1]
    int cnt = 0;
    for (int i = 0; i < m; ++i) {
        cnt += (poly[static_cast<size_t>(i)] != 0);
    }
    if (cnt == w) {
        return true;
    }

    // Circular sliding window
    for (int start = 1; start < r; ++start) {
        int out_idx = start - 1;
        int in_idx  = (start + m - 1) % r;

        cnt -= (poly[static_cast<size_t>(out_idx)] != 0);
        cnt += (poly[static_cast<size_t>(in_idx)] != 0);

        if (cnt == w) {
            return true;
        }
    }

    return false;
}

// --------------------------------------------------
// Determine whether it is a (m, epsilon=1)-cluster (in a looping sense)
// There exists a loop window of length m, which contains exactly w-1 ones,
// and there is exactly 1 one outside the window.
// --------------------------------------------------
bool is_m_cluster_poly_1_cyclic(const std::vector<uint8_t>& poly, int w, int m) {
    const int r = static_cast<int>(poly.size());

    if (r <= 0) {
        throw std::runtime_error("is_m_cluster_poly_1_cyclic: r must be positive");
    }
    if (m <= 0 || m > r) {
        throw std::runtime_error("is_m_cluster_poly_1_cyclic: m must satisfy 1 <= m <= r");
    }

    // Check if the total weight is w
    int total = 0;
    for (int i = 0; i < r; ++i) {
        total += (poly[static_cast<size_t>(i)] != 0);
    }
    if (total != w) {
        return false;
    }

    // The first window [0, m-1]
    int cnt = 0;
    for (int i = 0; i < m; ++i) {
        cnt += (poly[static_cast<size_t>(i)] != 0);
    }
    if (cnt == w - 1) {
        return true;
    }

    // Circular sliding window
    for (int start = 1; start < r; ++start) {
        int out_idx = start - 1;
        int in_idx  = (start + m - 1) % r;

        cnt -= (poly[static_cast<size_t>(out_idx)] != 0);
        cnt += (poly[static_cast<size_t>(in_idx)] != 0);

        if (cnt == w - 1) {
            return true;
        }
    }

    return false;
}


// --------------------------------------------------
// Enumerate all the i in the range [1, r-1] that are coprime with r
// --------------------------------------------------
std::vector<int> get_all_coprime_i(int r) {
    if (r <= 1) {
        throw std::runtime_error("get_all_coprime_i: r must be > 1");
    }

    std::vector<int> coprimes;
    for (int i = 1; i < r; ++i) {
        if (std::gcd(i, r) == 1) {
            coprimes.push_back(i);
        }
    }
    return coprimes;
}

// --------------------------------------------------
// Determine whether poly belongs to "(m, epsilon=0)-cluster after applying Phi"
// That is: does there exist gcd(i,r)=1 such that Phi_i(poly) is an ordinary (m,0)-cluster
//
// Note: Since the coprime multiplication modulo r forms a group, enumerating Phi_i(poly) or Phi_{i^{-1}}(poly)
// is essentially equivalent. Here we just directly enumerate Phi_i(poly).
// --------------------------------------------------
bool is_m_cluster_poly_0_phi_orbit(const std::vector<uint8_t>& poly, int w, int m) {
    const int r = static_cast<int>(poly.size());
    if (r <= 0) {
        throw std::runtime_error("is_m_cluster_poly_0_phi_orbit: r must be positive");
    }

    std::vector<int> coprimes = get_all_coprime_i(r);

    for (int i : coprimes) {
        std::vector<uint8_t> transformed = apply_phi_i(poly, r, i);
        if (is_m_cluster_poly_0_cyclic(transformed, w, m)) {
            return true;
        }
    }
    return false;
}

// --------------------------------------------------
// Determine whether poly belongs to “(m, epsilon=1)-cluster after applying Phi”
// That is: does there exist gcd(i,r)=1 such that Phi_i(poly) is an ordinary (m,1)-cluster
//
// Here we follow your previous definition of “exact epsilon=1”:
// exactly w-1 ones in the window, and exactly 1 one outside the window
// --------------------------------------------------
bool is_m_cluster_poly_1_phi_orbit(const std::vector<uint8_t>& poly, int w, int m) {
    const int r = static_cast<int>(poly.size());
    if (r <= 0) {
        throw std::runtime_error("is_m_cluster_poly_1_phi_orbit: r must be positive");
    }

    std::vector<int> coprimes = get_all_coprime_i(r);

    for (int i : coprimes) {
        std::vector<uint8_t> transformed = apply_phi_i(poly, r, i);
        if (is_m_cluster_poly_1_cyclic(transformed, w, m)) {
            return true;
        }
    }
    return false;
}

// --------------------------------------------------
// Generate: filter out random keys that are “(m, epsilon=0)-cluster after applying Phi”
//
// Note: This already automatically includes the filtering of ordinary “(m,0)-cluster” without Phi,
// because i=1 is also in the coprime set.
// --------------------------------------------------
std::vector<uint8_t> generate_random_poly_not_m_cluster_0_phi(int r,
                                                              int w,
                                                              int m,
                                                              const std::string& seed) {
    if (r <= 0) {
        throw std::runtime_error("generate_random_poly_not_m_cluster_0_phi: r must be positive");
    }
    if (w < 0 || w > r) {
        throw std::runtime_error("generate_random_poly_not_m_cluster_0_phi: invalid w");
    }
    if (m <= 0 || m > r) {
        throw std::runtime_error("generate_random_poly_not_m_cluster_0_phi: invalid m");
    }

    std::uint64_t attempt = 0;

    while (true) {
        std::string try_seed = seed + "|not_cluster_0_phi|try=" + std::to_string(attempt);
        std::vector<uint8_t> poly = generate_random_poly(r, w, try_seed);

        if (!is_m_cluster_poly_0_phi_orbit(poly, w, m)) {
            return poly;
        }

        ++attempt;
    }
}

// --------------------------------------------------
// Generate: filter out random keys that are “(m, epsilon=1)-cluster after applying Phi”
//
// Similarly, this only filters Phi-orbits with exact epsilon = 1.
// If you also want to exclude epsilon = 0, you need an additional combined judgment.
// --------------------------------------------------
std::vector<uint8_t> generate_random_poly_not_m_cluster_1_phi(int r,
                                                              int w,
                                                              int m,
                                                              const std::string& seed) {
    if (r <= 0) {
        throw std::runtime_error("generate_random_poly_not_m_cluster_1_phi: r must be positive");
    }
    if (w < 0 || w > r) {
        throw std::runtime_error("generate_random_poly_not_m_cluster_1_phi: invalid w");
    }
    if (m <= 0 || m > r) {
        throw std::runtime_error("generate_random_poly_not_m_cluster_1_phi: invalid m");
    }

    std::uint64_t attempt = 0;

    while (true) {
        std::string try_seed = seed + "|not_cluster_1_phi|try=" + std::to_string(attempt);
        std::vector<uint8_t> poly = generate_random_poly(r, w, try_seed);

        if (!is_m_cluster_poly_1_phi_orbit(poly, w, m)) {
            return poly;
        }

        ++attempt;
    }
}