#pragma once

#include <vector>
#include <cstdint>
#include <algorithm>

std::vector<uint8_t> rotate_right(const std::vector<uint8_t>& v, int l);

int weight_u8(const std::vector<uint8_t>& v);

int threshold(int S);

int bike_threshold(int S, int i, int S0, int w);

int ctr(const std::vector<uint8_t>& h_i,
        const std::vector<uint8_t>& s,
        int j);

struct BFIterResult {
    std::vector<uint8_t> e;     
    std::vector<uint8_t> black; 
    std::vector<uint8_t> gray;  
};

BFIterResult BFIter(const std::vector<uint8_t>& s,
                    const std::vector<uint8_t>& e_in,
                    int T, int tau,
                    const std::vector<std::vector<uint8_t>>& H);

std::vector<uint8_t> BFMaskedIter(const std::vector<uint8_t>& s,
                                  const std::vector<uint8_t>& e_in,
                                  const std::vector<uint8_t>& mask,
                                  int T,
                                  const std::vector<std::vector<uint8_t>>& H);

std::vector<uint8_t> matrix_multicative(const std::vector<uint8_t>& e,
                                        const std::vector<std::vector<uint8_t>>& H);

struct BGFDecodeResult {
    bool ok;
    std::vector<uint8_t> e;
};

BGFDecodeResult BGF(const std::vector<uint8_t>& s,
                    const std::vector<std::vector<uint8_t>>& H,
                    int w);

std::vector<uint8_t> bike_BFIter(const std::vector<uint8_t>& s,
                                 const std::vector<uint8_t>& e_in,
                                 int T,
                                 const std::vector<std::vector<uint8_t>>& H);

struct BikeDecodeResult {
    bool ok;
    std::vector<uint8_t> e;
};

BikeDecodeResult bike_decode(const std::vector<uint8_t>& s,
                             const std::vector<std::vector<uint8_t>>& H,
                             int w);

struct DecoderPrecomp {
    int r = 0;
    std::vector<int> h_support[2];
};

DecoderPrecomp build_decoder_precomp(const std::vector<std::vector<uint8_t>>& H);

struct DecoderScratch {
    std::vector<uint8_t> e;      
    std::vector<uint8_t> eH;     
    std::vector<uint8_t> s0;     
    std::vector<uint8_t> syn;    
    std::vector<uint8_t> black;  
    std::vector<uint8_t> gray;   

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

int ctr_support(const std::vector<int>& support,
                const uint8_t* s,
                int j,
                int r);

void extract_error_support(const std::vector<uint8_t>& e,
                           int r,
                           std::vector<int>& e0_support,
                           std::vector<int>& e1_support);

void matrix_multicative_fast(const std::vector<uint8_t>& e,
                             const DecoderPrecomp& pc,
                             std::vector<uint8_t>& s_out,
                             std::vector<int>& e0_support,
                             std::vector<int>& e1_support);

void matrix_multicative_first_half_from_positions(
    const std::vector<int>& ones_pos,
    const DecoderPrecomp& pc,
    std::vector<uint8_t>& s_out);

bool BGF_check_fast(const DecoderPrecomp& pc,
                    int w,
                    DecoderScratch& ws);

bool bike_decode_check_fast(const DecoderPrecomp& pc,
                            int w,
                            DecoderScratch& ws);