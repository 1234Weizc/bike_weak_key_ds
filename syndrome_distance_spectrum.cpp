#include <vector>
#include <fstream>
#include <utility>
#include <string>
#include <algorithm>
#include <set>
#include <cstdint>
#include <chrono> 
#include <iomanip>
#include <cmath>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>
#include <atomic>
#include <sstream>
#include <limits>
#include <stdexcept>
#include <random>
#include <chrono>

#include <openssl/evp.h>
#include <openssl/sha.h>

#include "SHAKE256.h"
#include "key_generator.h"
#include "decoder.h"

// =========================
// parameters
// =========================

// level 1
int r = 12323; // Block length
int w = 142;   // weight
int t = 134;   // weight of error vector e 
int d = w / 2;

constexpr std::uint64_t PROGRESS_FLUSH_BATCH = 256;

// Do you want to print the logs sample by sample?
constexpr bool VERBOSE_PER_SAMPLE_LOG = false;

// Print the progress every time a certain number of samples are processed.
constexpr std::uint64_t PROGRESS_REPORT_EVERY = 10000;

// =========================
// Thread-safe Log
// =========================
std::mutex g_log_mutex;

void safe_log(const std::string& msg) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    std::cout << msg;
}

// =========================
// Distance Spectrum of Keys: Used to store the distance spectrum of h
// =========================
std::pair<std::set<int>, std::vector<std::uint64_t>>
compute_distance_spectrum(const std::vector<uint8_t>& h)
{
    int r_local = static_cast<int>(h.size());
    std::vector<int> ones_pos;
    ones_pos.reserve(r_local);

    for (int i = 0; i < r_local; ++i) {
        if (h[i] == 1) {
            ones_pos.push_back(i);
        }
    }

    int half_r = r_local / 2;
    std::set<int> DS;
    std::vector<std::uint64_t> count(half_r, 0);

    for (size_t i = 0; i < ones_pos.size(); ++i) {
        for (size_t j = i + 1; j < ones_pos.size(); ++j) {
            int x = ones_pos[i];
            int y = ones_pos[j];
            int dist = std::min((x - y + r_local) % r_local, (y - x + r_local) % r_local);
            DS.insert(dist);
            count[dist - 1]++;
        }
    }

    return {DS, count};
}

// =========================
// Calculate syndrome weight
// =========================
double compute_syndrome_weight_from_positions(
    const std::vector<int>& ones_pos,
    const DecoderPrecomp& H_pre,
    DecoderScratch& ws)
{
    const int r = H_pre.r;
    const int n = 2 * r;

    std::fill(ws.e.begin(), ws.e.end(), 0);

    for (int pos : ones_pos) {
        if (pos < 0 || pos >= n) {
            throw std::runtime_error("compute_syndrome_weight: error position out of range");
        }
        ws.e[static_cast<size_t>(pos)] = 1;
    }

    // Calculate syndrome s = e0*h0 + e1*h1
    matrix_multicative_fast(ws.e, H_pre, ws.s0, ws.e0_support, ws.e1_support);

    // Calculate wt(s)
    int wt = 0;
    for (uint8_t x : ws.s0) {
        wt += x;
    }

    // Add normal noise N(0, sigma^2)
    static thread_local std::mt19937_64 rng(
        static_cast<unsigned long long>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count()
        ) ^ std::hash<std::thread::id>{}(std::this_thread::get_id())
    );
    double sigma = 0;
    std::normal_distribution<double> noise_dist(0.0, sigma);
    double noise = noise_dist(rng);

    return static_cast<double>(wt) + noise;
}

// =========================
// Thread-local statistics structure
// =========================
struct ThreadLocalStats {
    std::vector<std::uint64_t> a;
    std::vector<std::uint64_t> b;

    explicit ThreadLocalStats(int half_r)
        : a(static_cast<size_t>(half_r), 0ULL),
          b(static_cast<size_t>(half_r), 0ULL) {}
};

struct WorkerScratch {
    std::vector<int> ones_pos;     // Error positions of total length 2r
    std::vector<int> e0_ones_pos;  // Error positions of the first half block e[0:r]
    DecoderScratch decoder_ws;

    WorkerScratch(int r_local, int t_local)
        : decoder_ws(r_local)
    {
        ones_pos.reserve(static_cast<size_t>(t_local));
        e0_ones_pos.reserve(static_cast<size_t>(t_local));
    }
};

// =========================
// Accumulate empirical distance spectrum contributions of each e_i into stats by multiplicity
// =========================
void accumulate_empirical_for_e0_positions(
    const std::vector<int>& e0_ones_pos,
    int r_local,
    int wt_s,
    ThreadLocalStats& stats)
{
    const int half_r = r_local / 2;
    std::vector<std::uint64_t> count(static_cast<size_t>(half_r), 0ULL);

    for (size_t i = 0; i < e0_ones_pos.size(); ++i) {
        const int x = e0_ones_pos[i];
        if (x < 0 || x >= r_local) {
            throw std::runtime_error("accumulate_empirical_for_e0_positions: e0 position out of range");
        }

        for (size_t j = i + 1; j < e0_ones_pos.size(); ++j) {
            const int y = e0_ones_pos[j];
            if (y < 0 || y >= r_local) {
                throw std::runtime_error("accumulate_empirical_for_e0_positions: e0 position out of range");
            }

            const int dist = std::min(
                (x - y + r_local) % r_local,
                (y - x + r_local) % r_local
            );

            if (dist <= 0 || dist > half_r) {
                throw std::runtime_error("accumulate_empirical_for_e0_positions: invalid cyclic distance");
            }

            count[static_cast<size_t>(dist - 1)]++;
        }
    }

    for (size_t idx = 0; idx < count.size(); ++idx) {
        const std::uint64_t v = count[idx];
        if (v == 0ULL) {
            continue;
        }

        stats.a[idx] += static_cast<std::uint64_t>(wt_s) * v;
        stats.b[idx] += v;
    }
}

// =========================
// Multi-threaded streaming generation and accumulation of empirical distance spectrum
// =========================
ThreadLocalStats generate_and_accumulate_empirical_spectrum_mt(
    int num,
    int r,
    int w,
    int t,
    const std::string& seed,
    const DecoderPrecomp& H_pre,
    unsigned int num_threads)
{
    (void)w;

    const int half_r = r / 2;
    const std::uint64_t total_samples = static_cast<std::uint64_t>(num);

    if (num_threads == 0) {
        num_threads = 1;
    }
    if (num_threads > static_cast<unsigned int>(num)) {
        num_threads = static_cast<unsigned int>(num == 0 ? 1 : num);
    }

    std::atomic<std::uint64_t> processed_samples{0};

    std::vector<ThreadLocalStats> locals;
    locals.reserve(num_threads);
    for (unsigned int tid = 0; tid < num_threads; ++tid) {
        locals.emplace_back(half_r);
    }

    std::vector<std::thread> workers;
    workers.reserve(num_threads);

    auto worker_func = [&](unsigned int tid, int begin_i, int end_i) {
        ThreadLocalStats& local = locals[tid];
        WorkerScratch scratch(r, t);

        std::uint64_t local_processed = 0;
        std::string local_seed;
        local_seed.reserve(seed.size() + 48);

        auto flush_progress = [&]() {
            if (local_processed == 0) return;

            std::uint64_t prev = processed_samples.fetch_add(local_processed);
            std::uint64_t done = prev + local_processed;
            bool crossed =
                (prev / PROGRESS_REPORT_EVERY) != (done / PROGRESS_REPORT_EVERY);

            local_processed = 0;

            if (!VERBOSE_PER_SAMPLE_LOG && (crossed || done == total_samples)) {
                std::ostringstream oss;
                oss << "[INFO] Progress: " << done << "/" << total_samples
                    << " (" << (100.0 * static_cast<double>(done) /
                                static_cast<double>(total_samples))
                    << "%)\n";
                safe_log(oss.str());
            }
        };

        for (int i = begin_i; i < end_i; ++i) {
            const std::string i_str = std::to_string(i);

            local_seed.clear();
            local_seed.append(seed)
                      .append("|rand_error_2r|")
                      .append(i_str);

            // =================================================================
            // // Generate a fully random list of error positions with length 2r and Hamming weight t
            std::vector<int> err_pos_2r = generate_wlist(2 * r, t, local_seed);

            // scratch.ones_pos:
            // Error positions over the full 2r-length range for syndrome computation
            scratch.ones_pos.clear();
            scratch.ones_pos.reserve(err_pos_2r.size());

            // scratch.e0_ones_pos:
            // Only retain the 1s in the first r positions for empirical distance spectrum statistics
            scratch.e0_ones_pos.clear();
            scratch.e0_ones_pos.reserve(err_pos_2r.size());

            for (int pos : err_pos_2r) {
                scratch.ones_pos.push_back(pos);

                if (pos < r) {
                    scratch.e0_ones_pos.push_back(pos);
                }
            }
            // =================================================================
            
            int wt_s = compute_syndrome_weight_from_positions(
                scratch.ones_pos,
                H_pre,
                scratch.decoder_ws
            );

            // The empirical distance spectrum is computed solely on e[0:r]
            accumulate_empirical_for_e0_positions(
                scratch.e0_ones_pos,
                r,
                wt_s,
                local
            );

            if (VERBOSE_PER_SAMPLE_LOG) {
                std::ostringstream oss;
                oss << "[INFO] Thread " << tid
                    << " generated random 2r sample i=" << (i + 1) << "/" << num
                    << ", wt_s=" << wt_s << "\n";
                safe_log(oss.str());
            }

            ++local_processed;
            if (local_processed >= PROGRESS_FLUSH_BATCH) {
                flush_progress();
            }
        }

        flush_progress();
    };

    const int chunk = num / static_cast<int>(num_threads);
    const int rem = num % static_cast<int>(num_threads);

    int current_begin = 0;
    for (unsigned int tid = 0; tid < num_threads; ++tid) {
        int size = chunk + (static_cast<int>(tid) < rem ? 1 : 0);
        int begin_i = current_begin;
        int end_i = begin_i + size;
        current_begin = end_i;

        workers.emplace_back(worker_func, tid, begin_i, end_i);
    }

    for (auto& th : workers) {
        th.join();
    }

    // Main thread merging
    ThreadLocalStats global_stats(half_r);
    for (unsigned int tid = 0; tid < num_threads; ++tid) {
        for (int i = 0; i < half_r; ++i) {
            global_stats.a[i] += locals[tid].a[i];
            global_stats.b[i] += locals[tid].b[i];
        }
    }

    return global_stats;
}

// =========================
// Write the statistical results of the experience distance spectrum to CSV
// =========================
void save_empirical_spectrum_to_csv(
    const ThreadLocalStats& stats,
    const std::string& filename)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open file " << filename << " for writing\n";
        return;
    }

    file << "Distance,AvgWtS,SumWt,SumMultiplicity\n";
    for (size_t i = 0; i < stats.a.size(); ++i) {
        double der = 0.0;
        if (stats.b[i] > 0) {
            der = static_cast<double>(stats.a[i]) / static_cast<double>(stats.b[i]);
        }
        file << (i + 1) << ","
             << der << ","
             << stats.a[i] << ","
             << stats.b[i] << "\n";
    }

    file.close();
    std::cout << "The empirical result is saved in " << filename << std::endl;
}

// =========================
// Save the distance spectrum of key h in CSV format
// =========================
void save_distance_spectrum_to_csv(
    const std::vector<uint8_t>& h,
    const std::string& filename)
{
    auto [DS, count] = compute_distance_spectrum(h);

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open file " << filename << " for writing\n";
        return;
    }

    file << "Distance,Count\n";
    for (size_t i = 0; i < count.size(); ++i) {
        file << (i + 1) << "," << count[i] << "\n";
    }

    file.close();
    std::cout << "The distance spectrum is saved in " << filename << std::endl;
}

// Save H(2) to a csv file
void save_H_to_csv(const std::vector<std::vector<uint8_t>>& H, const std::string& filename) {
    if (H.size() != 2) {
        throw std::runtime_error("save_H_to_csv: H.size() != 2");
    }
    if (H[0].size() != H[1].size()) {
        throw std::runtime_error("save_H_to_csv: H[0].size() != H[1].size()");
    }

    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        throw std::runtime_error("save_H_to_csv: cannot open file " + filename);
    }

    ofs << "block";
    for (size_t i = 0; i < H[0].size(); ++i) {
        ofs << ",pos_" << i;
    }
    ofs << '\n';

    ofs << "H0";
    for (size_t i = 0; i < H[0].size(); ++i) {
        ofs << "," << static_cast<int>(H[0][i]);
    }
    ofs << '\n';

    ofs << "H1";
    for (size_t i = 0; i < H[1].size(); ++i) {
        ofs << "," << static_cast<int>(H[1][i]);
    }
    ofs << '\n';

    ofs.close();
}

static std::string trim_copy(const std::string& s) {
    size_t l = 0, r = s.size();

    while (l < r && (s[l] == ' ' || s[l] == '\t' || s[l] == '\r' || s[l] == '\n')) ++l;
    while (r > l && (s[r - 1] == ' ' || s[r - 1] == '\t' || s[r - 1] == '\r' || s[r - 1] == '\n')) --r;

    std::string t = s.substr(l, r - l);

    if (t.size() >= 3 &&
        static_cast<unsigned char>(t[0]) == 0xEF &&
        static_cast<unsigned char>(t[1]) == 0xBB &&
        static_cast<unsigned char>(t[2]) == 0xBF) {
        t = t.substr(3);
    }

    return t;
}

// =========================
// Extract the list of positions where h has value 1 (for saving key row)
// =========================
std::vector<int> extract_support_positions(const std::vector<uint8_t>& h) {
    std::vector<int> pos;
    pos.reserve(h.size());

    for (int i = 0; i < static_cast<int>(h.size()); ++i) {
        if (h[static_cast<size_t>(i)] == 1) {
            pos.push_back(i);
        }
    }
    return pos;
}

// =========================
// Construct one row of AvgWtS from empirical distance spectrum statistics
// To be compatible with num_dist = r/2 + 1 on the Python side,
// pad 0.0 at index 0, and place actual distances 1..r/2 in row[1..r/2]
// =========================
std::vector<double> build_avgwts_row_with_zero_slot(const ThreadLocalStats& stats) {
    const int half_r = static_cast<int>(stats.a.size());

    // row[0] corresponds to "distance 0" and is fixed to 0.0
    std::vector<double> row(static_cast<size_t>(half_r + 1), 0.0);

    for (int dist = 1; dist <= half_r; ++dist) {
        const size_t idx = static_cast<size_t>(dist - 1);

        if (stats.b[idx] > 0ULL) {
            row[static_cast<size_t>(dist)] =
                static_cast<double>(stats.a[idx]) / static_cast<double>(stats.b[idx]);
        } else {
            row[static_cast<size_t>(dist)] = 0.0;
        }
    }

    return row;
}

// =========================
// Initialize and clear the output trace file
// Data will be appended in the format: key, DS, key, DS, ...
// =========================
void reset_trace_output_file(const std::string& filename) {
    std::ofstream file(filename, std::ios::trunc);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file " + filename + " for resetting");
    }
    file.close();
}

// =========================
// Append a pair (key, AvgWtS) to the trace CSV file
// key row: positions of 1s in h0 (integers)
// DS  row: corresponding AvgWtS (floating-point numbers)
// No header is written, directly compatible with Python's read_trace_file()
// =========================
void append_key_and_avgwts_to_trace_csv(
    const std::vector<uint8_t>& h0,
    const ThreadLocalStats& stats,
    const std::string& filename)
{
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file " + filename + " for appending");
    }

    // -------- Line 1: key = support positions of h0 --------
    std::vector<int> key_positions = extract_support_positions(h0);

    for (size_t i = 0; i < key_positions.size(); ++i) {
        if (i > 0) file << ",";
        file << key_positions[i];
    }
    file << "\n";

    // -------- Line 2: DS = AvgWtS row --------
    std::vector<double> avgwts_row = build_avgwts_row_with_zero_slot(stats);

    file << std::setprecision(17);
    for (size_t i = 0; i < avgwts_row.size(); ++i) {
        if (i > 0) file << ",";
        file << avgwts_row[i];
    }
    file << "\n";

    file.close();
}

std::vector<std::vector<uint8_t>> load_H_from_csv(const std::string& filename, int r) {
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        throw std::runtime_error("Cannot open csv file: " + filename);
    }

    std::string line;
    if (!std::getline(fin, line)) {
        throw std::runtime_error("CSV file is empty: " + filename);
    }

    std::vector<std::vector<uint8_t>> H(2, std::vector<uint8_t>(static_cast<size_t>(r), 0));
    bool found_H0 = false;
    bool found_H1 = false;

    while (std::getline(fin, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> cells;

        while (std::getline(ss, cell, ',')) {
            cells.push_back(trim_copy(cell));
        }

        if (cells.empty()) continue;

        std::string tag = cells[0];
        int idx = -1;

        if (tag == "H0") idx = 0;
        else if (tag == "H1") idx = 1;
        else continue;

        if (static_cast<int>(cells.size()) != r + 1) {
            throw std::runtime_error(
                "CSV row length mismatch in " + tag +
                ": expected " + std::to_string(r + 1) +
                ", got " + std::to_string(cells.size())
            );
        }

        for (int i = 0; i < r; ++i) {
            const std::string& x = cells[static_cast<size_t>(i + 1)];
            if (x == "0") H[idx][static_cast<size_t>(i)] = 0;
            else if (x == "1") H[idx][static_cast<size_t>(i)] = 1;
            else {
                throw std::runtime_error(
                    "Invalid cell value in " + tag +
                    " at position " + std::to_string(i) +
                    ": " + x
                );
            }
        }

        if (idx == 0) found_H0 = true;
        if (idx == 1) found_H1 = true;
    }

    if (!found_H0 || !found_H1) {
        throw std::runtime_error("CSV does not contain both H0 and H1 rows.");
    }

    return H;
}

int main() {
    auto program_start = std::chrono::steady_clock::now();

    int f = 30; // type 1 weak key block length f
    int m = 5100; // Error and key clustering degree
    (void)f;
    (void)m;

    int num = 10000;   // Number of error samples corresponding to each H
    int main_num = 10000;    // Number of experiments: generate main_num different H=(h0,h1)

    std::string seed = "random_seed";

    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()
    ).count();

    seed = seed + "_" + std::to_string(ms);

    std::cout << "r = " << r
              << ", w = " << w
              << ", t = " << t
              << ", d = " << d << std::endl;
    std::cout << "num = " << num
              << ", main_num = " << main_num << std::endl;

    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) {
        num_threads = 4;
    }

    std::cout << "Using " << num_threads << " threads." << std::endl;

    // Unified output file: append data in the format: key, DS, key, DS, ...
    const std::string trace_filename = "test_trace.csv";
    reset_trace_output_file(trace_filename);

    for (int main_idx = 1; main_idx <= main_num; ++main_idx) {
        auto one_exp_start = std::chrono::steady_clock::now();

        std::cout << "\n==============================" << std::endl;
        std::cout << "Begin experiment " << main_idx
                  << " / " << main_num << std::endl;
        std::cout << "==============================" << std::endl;

        // Generate an independent seed for each experimental run
        std::string exp_seed = seed + "|main_idx=" + std::to_string(main_idx);

        // generate H=(h0,h1)
        std::vector<std::vector<uint8_t>> H(2);
        H[0] = generate_random_poly(r, d, exp_seed + "|H[0]");
        H[1] = generate_random_poly(r, d, exp_seed + "|H[1]");

        // Preprocessing
        DecoderPrecomp H_pre = build_decoder_precomp(H);

        std::cout << "Begin to compute empirical spectrum for experiment "
                  << main_idx << " ..." << std::endl;

        // Compute empirical distance spectrum statistics
        ThreadLocalStats stats = generate_and_accumulate_empirical_spectrum_mt(
            num, r, w, t, exp_seed, H_pre, num_threads
        );

        // Append data to the global file in two-line format: key followed by DS
        // Note: This saves the support positions of H[0],
        // compatible with get_train_labels_from_key(wlist) on the Python side.
        append_key_and_avgwts_to_trace_csv(H[0], stats, trace_filename);

        std::cout << "Appended experiment " << main_idx
                  << " to " << trace_filename << std::endl;

        auto one_exp_end = std::chrono::steady_clock::now();
        std::chrono::duration<double> one_exp_elapsed = one_exp_end - one_exp_start;

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Experiment " << main_idx
                  << " finished in "
                  << one_exp_elapsed.count()
                  << " seconds." << std::endl;
    }

    std::cout << "\nAll experiments done!" << std::endl;

    auto program_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = program_end - program_start;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Total execution time: "
              << elapsed_seconds.count()
              << " seconds." << std::endl;

    return 0;
}