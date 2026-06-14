#pragma once

EVP_MD_CTX* prng_init_shake256(const std::string& seed);

int randint_uniform(int n, EVP_MD_CTX* ctx);