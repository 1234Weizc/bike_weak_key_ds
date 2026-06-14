#include <iostream>
#include <openssl/evp.h>  
#include <openssl/sha.h>  
#include <string>  
#include <vector>   

// Correct initialization of the SHAKE256 PRNG
EVP_MD_CTX* prng_init_shake256(const std::string& seed) {
    // Create and initialize the SHAKE256 context
    EVP_MD_CTX* ctx = EVP_MD_CTX_new(); 
    const EVP_MD* md = EVP_shake256(); 
    EVP_DigestInit_ex(ctx, md, NULL); 
    
    // Input seed
    EVP_DigestUpdate(ctx, seed.c_str(), seed.length());
    
    return ctx; // Return the context for subsequent generation of random numbers
}

// Simulate uniform random (F-Y alg.)
int randint_uniform(int n, EVP_MD_CTX* ctx) {
    const uint64_t limit = (uint64_t(1) << 32);
    const uint64_t bound = (limit / uint32_t(n)) * uint32_t(n);

    while (true) {
        unsigned char b[4];
        EVP_DigestFinalXOF(ctx, b, 4); 
        // Each time, a new 4-byte value is obtained and it is mapped to b[i], which represents one byte (8 bits).

        uint32_t x32 =
            (uint32_t)b[0]
          | ((uint32_t)b[1] << 8)
          | ((uint32_t)b[2] << 16)
          | ((uint32_t)b[3] << 24);
        // Combine four bytes into a 32-bit unsigned integer through bitwise operations
        if ((uint64_t)x32 < bound) return int(x32 % uint32_t(n)); 
        // If the generated number is within the acceptable range, return the result of taking it modulo n
    }
}


// test code
// int main() {
//     std::string seed = std::to_string(time(nullptr)); // 用当前时间戳作为种子
//     EVP_MD_CTX* ctx = prng_init_shake256(seed);

//     for (int i = 0; i < 10; i++) {
//         int r = randint_uniform(100, ctx);
//         std::cout << "Random number " << i << ": " << r << "\n";
//     }

//     EVP_MD_CTX_free(ctx); // 释放上下文资源
// }