// key_generator.h
#ifndef KEY_GENERATOR_H
#define KEY_GENERATOR_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

std::vector<int> generate_wlist(int r, int w, const std::string& seed);

std::vector<uint8_t> generate_random_poly(int r, int w, const std::string& seed);

std::vector<uint8_t> generate_weak1_poly(int r, int w, int f, const std::string& seed);

std::vector<uint8_t> generate_weak2_poly(
    int r, int w, int m,
    const std::string& seed_a,
    const std::string& seed_b
);

std::pair<std::vector<uint8_t>, std::vector<uint8_t>> generate_weak3_poly(
    int r, int d, int m, const std::string& seed_base
);

std::vector<uint8_t> generate_m_cluster_poly_0(int r, int w, int m, const std::string& seed);

std::vector<uint8_t> generate_m_cluster_poly_1(int r, int w, int m, const std::string& seed);

std::pair<std::vector<uint8_t>, std::vector<uint8_t>> generate_flip_intersection_key(
    int r, int d, int m, const std::string& seed_base
);

// Tool
std::vector<uint8_t> rotate_right(const std::vector<uint8_t>& v, int l);
int weight_u8(const std::vector<uint8_t>& e);

std::vector<uint8_t> apply_phi_i(const std::vector<uint8_t>& f, int r, int i);

int generate_random_coprime_i(int r, const std::string& seed);

std::vector<uint8_t> generate_m_cluster_poly_0_phi(int r, int w, int m, const std::string& seed);

std::vector<uint8_t> generate_m_cluster_poly_1_phi(int r, int w, int m, const std::string& seed);

bool is_m_cluster_poly_0_cyclic(const std::vector<uint8_t>& poly, int w, int m);

bool is_m_cluster_poly_1_cyclic(const std::vector<uint8_t>& poly, int w, int m);

std::vector<int> get_all_coprime_i(int r);

bool is_m_cluster_poly_0_phi_orbit(const std::vector<uint8_t>& poly, int w, int m);

bool is_m_cluster_poly_1_phi_orbit(const std::vector<uint8_t>& poly, int w, int m);

std::vector<uint8_t> generate_random_poly_not_m_cluster_0_phi(int r,
                                                              int w,
                                                              int m,
                                                              const std::string& seed);

std::vector<uint8_t> generate_random_poly_not_m_cluster_1_phi(int r,
                                                              int w,
                                                              int m,
                                                              const std::string& seed);

#endif