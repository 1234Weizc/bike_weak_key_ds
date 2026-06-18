#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

struct SearchOptions {
    std::uint64_t max_nodes = 0;              // 0 Indicates no restriction
    std::uint64_t max_candidate_checks = 0;   // 0 Indicates no restriction
};

struct SearchResult {
    bool found = false;
    int found_shift = -1;  // if found=true，Indicate f(x)=x^found_shift h0(x)

    std::uint64_t visited_nodes = 0;
    std::uint64_t checked_candidates = 0;

    bool stopped_by_node_limit = false;
    bool stopped_by_candidate_limit = false;
};

struct Instance {
    std::vector<uint8_t> h0;
    std::vector<int> D;
    std::size_t pair_id = 0;
};

int mod_pos(int x, int r) {
    int y = x % r;
    if (y < 0) y += r;
    return y;
}

int normalize_distance(int d, int r) {
    int x = mod_pos(d, r);
    return std::min(x, r - x);
}

int hamming_weight(const std::vector<uint8_t>& v) {
    int wt = 0;
    for (uint8_t x : v) {
        if (x) ++wt;
    }
    return wt;
}

std::vector<int> support_of_binary_vector(const std::vector<uint8_t>& v) {
    std::vector<int> supp;

    for (int i = 0; i < static_cast<int>(v.size()); ++i) {
        if (v[static_cast<std::size_t>(i)] != 0) {
            supp.push_back(i);
        }
    }

    return supp;
}

// Determine whether the position pos equals 1 under shift k
// f(x) = x^k h0(x)
// Thus f[pos] = h0[(pos - k) mod r]
static inline uint8_t shifted_h0_value(
    const std::vector<uint8_t>& h0,
    int r,
    int k,
    int pos
) {
    int src = mod_pos(pos - k, r);
    return h0[static_cast<std::size_t>(src)] ? 1 : 0;
}

// Initialize all possible cyclic shifts k.
// The positions that are already 1 in the current initial f must all fall on the 1 positions of x^k h0.
std::vector<int> initial_viable_shifts(
    const std::vector<uint8_t>& f,
    const std::vector<uint8_t>& h0,
    int r
) {
    std::vector<int> shifts;

    for (int k = 0; k < r; ++k) {
        bool ok = true;

        for (int pos = 0; pos < r; ++pos) {
            if (f[static_cast<std::size_t>(pos)] != 0) {
                if (shifted_h0_value(h0, r, k, pos) == 0) {
                    ok = false;
                    break;
                }
            }
        }

        if (ok) {
            shifts.push_back(k);
        }
    }

    return shifts;
}

// Filter out valid cyclic shifts after assigning f[pos] = value once.
// If value = 1, x^k h0 must also equal 1 at position pos.
// If value = 0, x^k h0 must also equal 0 at position pos.
std::vector<int> filter_shifts_by_assignment(
    const std::vector<int>& old_shifts,
    const std::vector<uint8_t>& h0,
    int r,
    int pos,
    int value
) {
    std::vector<int> new_shifts;
    new_shifts.reserve(old_shifts.size());

    uint8_t v = static_cast<uint8_t>(value != 0);

    for (int k : old_shifts) {
        if (shifted_h0_value(h0, r, k, pos) == v) {
            new_shifts.push_back(k);
        }
    }

    return new_shifts;
}

// Extract all integers from a single line of CSV or space-separated text
std::vector<int> parse_ints_from_line(const std::string& line) {
    std::string tmp = line;

    for (char& ch : tmp) {
        unsigned char c = static_cast<unsigned char>(ch);
        if (!(std::isdigit(c) || ch == '-' || ch == '+')) {
            ch = ' ';
        }
    }

    std::stringstream ss(tmp);
    std::vector<int> nums;

    long long x;
    while (ss >> x) {
        if (x < std::numeric_limits<int>::min() ||
            x > std::numeric_limits<int>::max()) {
            throw std::runtime_error("integer out of int range in CSV line");
        }
        nums.push_back(static_cast<int>(x));
    }

    return nums;
}

// Two formats are supported for h0 rows:
// 1. Binary 0/1 vector of length r: 0,1,0,1,...
// 2. List of positions with value 1: 0,1,2,4800,...
std::vector<uint8_t> parse_h0_line(const std::string& line, int r) {
    std::vector<int> nums = parse_ints_from_line(line);

    if (nums.empty()) {
        throw std::runtime_error("empty h0 line");
    }

    std::vector<uint8_t> h0(static_cast<std::size_t>(r), 0);

    bool looks_like_binary_vector =
        static_cast<int>(nums.size()) == r &&
        std::all_of(nums.begin(), nums.end(), [](int x) {
            return x == 0 || x == 1;
        });

    if (looks_like_binary_vector) {
        for (int i = 0; i < r; ++i) {
            h0[static_cast<std::size_t>(i)] =
                static_cast<uint8_t>(nums[static_cast<std::size_t>(i)]);
        }
        return h0;
    }

    // Otherwise parse as the support position list
    for (int p : nums) {
        if (p < 0 || p >= r) {
            throw std::runtime_error("h0 support position out of range");
        }
        h0[static_cast<std::size_t>(p)] = 1;
    }

    return h0;
}

std::vector<int> parse_D_line(const std::string& line) {
    return parse_ints_from_line(line);
}

// Read CSV: by default rows 1, 3, 5, ... correspond to D; rows 2, 4, 6, ... correspond to h0
std::vector<Instance> load_instances_from_csv(
    const std::string& filename,
    int r,
    bool first_line_is_D
) {
    std::ifstream in(filename);
    if (!in) {
        throw std::runtime_error("cannot open input CSV file");
    }

    std::vector<std::string> useful_lines;
    std::string line;

    while (std::getline(in, line)) {
        std::vector<int> nums = parse_ints_from_line(line);

        // Skip blank lines or lines containing only headers
        if (!nums.empty()) {
            useful_lines.push_back(line);
        }
    }

    if (useful_lines.size() % 2 != 0) {
        throw std::runtime_error(
            "CSV useful line count must be even: every pair should contain D and h0"
        );
    }

    std::vector<Instance> instances;

    for (std::size_t i = 0; i < useful_lines.size(); i += 2) {
        const std::string& line1 = useful_lines[i];
        const std::string& line2 = useful_lines[i + 1];

        Instance inst;
        inst.pair_id = instances.size();

        if (first_line_is_D) {
            inst.D = parse_D_line(line1);
            inst.h0 = parse_h0_line(line2, r);
        } else {
            inst.h0 = parse_h0_line(line1, r);
            inst.D = parse_D_line(line2);
        }

        instances.push_back(std::move(inst));
    }

    return instances;
}

// Order: place 15, 16, 17, 18, 19 at the front, then append entries in D's original order
std::vector<int> build_order(const std::vector<int>& D, int r) {
    std::vector<int> order;
    std::unordered_set<int> seen;

    auto add_distance = [&](int raw_d) {
        int d = normalize_distance(raw_d, r);

        if (d == 0) return;

        int max_d = r / 2;
        if (d > max_d) return;

        if (seen.insert(d).second) {
            order.push_back(d);
        }
    };

    for (int d = 15; d <= 19; ++d) {
        add_distance(d);
    }

    for (int d : D) {
        add_distance(d);
    }

    return order;
}

bool apply_value(
    std::vector<uint8_t>& f,
    int pos,
    int value,
    int& weight,
    std::vector<int>& changed,
    std::vector<int>& supp_f
) {
    if (value == 0) {
        return f[static_cast<std::size_t>(pos)] == 0;
    }

    if (value == 1) {
        if (f[static_cast<std::size_t>(pos)] == 0) {
            f[static_cast<std::size_t>(pos)] = 1;
            changed.push_back(pos);
            supp_f.push_back(pos);
            ++weight;
        }
        return true;
    }

    throw std::runtime_error("value must be 0 or 1");
}

void rollback(
    std::vector<uint8_t>& f,
    int& weight,
    const std::vector<int>& changed,
    std::vector<int>& supp_f
) {
    for (int pos : changed) {
        f[static_cast<std::size_t>(pos)] = 0;
        --weight;
    }

    supp_f.resize(supp_f.size() - changed.size());
}

// Check if there exists an integer k such that
// supp_f = { (p + k) mod r | p \in supp_h0 }
//
// Equivalently, f(x) ≡ x^k h0(x) mod (x^r − 1)
bool equal_up_to_cyclic_shift_by_support(
    const std::vector<int>& supp_f,
    const std::vector<int>& supp_h0,
    int r,
    int& found_shift
) {
    found_shift = -1;

    if (supp_f.size() != supp_h0.size()) {
        return false;
    }

    if (supp_f.empty()) {
        found_shift = 0;
        return true;
    }

    std::unordered_set<int> f_set;
    f_set.reserve(supp_f.size() * 2 + 10);

    for (int p : supp_f) {
        f_set.insert(mod_pos(p, r));
    }

    // Determine candidate values of k using one anchor point of f
    int anchor_f = mod_pos(supp_f[0], r);

    std::unordered_set<int> tried_shift;
    tried_shift.reserve(supp_h0.size() * 2 + 10);

    for (int p0 : supp_h0) {
        int k = mod_pos(anchor_f - p0, r);

        if (!tried_shift.insert(k).second) {
            continue;
        }

        bool ok = true;

        for (int p : supp_h0) {
            int shifted_pos = mod_pos(p + k, r);

            if (!f_set.count(shifted_pos)) {
                ok = false;
                break;
            }
        }

        if (ok) {
            found_shift = k;
            return true;
        }
    }

    return false;
}

bool dfs_find_target_shift_pruned(
    const std::vector<int>& order,
    const std::vector<int>& suffix_max_add,
    int idx,
    int r,
    int target_weight,
    std::vector<uint8_t>& f,
    int& weight,
    const std::vector<uint8_t>& target_h0,
    std::vector<int>& supp_f,
    const std::vector<int>& viable_shifts,
    SearchResult& result,
    const SearchOptions& opt
) {
    if (result.stopped_by_node_limit ||
        result.stopped_by_candidate_limit ||
        result.found) {
        return result.found;
    }

    if (viable_shifts.empty()) {
        return false;
    }

    if (opt.max_nodes != 0 && result.visited_nodes >= opt.max_nodes) {
        result.stopped_by_node_limit = true;
        return false;
    }

    ++result.visited_nodes;

    if (weight == target_weight) {
        if (opt.max_candidate_checks != 0 &&
            result.checked_candidates >= opt.max_candidate_checks) {
            result.stopped_by_candidate_limit = true;
            return false;
        }

        ++result.checked_candidates;

        // As long as feasible shifts remain, the current f equals some x^k h0
        result.found = true;
        result.found_shift = viable_shifts[0];
        return true;
    }

    if (weight > target_weight) {
        return false;
    }

    if (idx >= static_cast<int>(order.size())) {
        return false;
    }

    if (weight + suffix_max_add[static_cast<std::size_t>(idx)] < target_weight) {
        return false;
    }

    int d = order[static_cast<std::size_t>(idx)];
    int pos_plus = mod_pos(d, r);
    int pos_minus = mod_pos(-d, r);

    const int cases[4][2] = {
        {1, 0},
        {0, 1},
        {0, 0},
        {1, 1}
    };

    for (const auto& c : cases) {
        if (result.stopped_by_node_limit ||
            result.stopped_by_candidate_limit ||
            result.found) {
            return result.found;
        }

        if (pos_plus == pos_minus && c[0] != c[1]) {
            continue;
        }

        std::vector<int> changed;
        bool ok = true;

        // Prune based on cyclic shifts first
        std::vector<int> shifts1 =
            filter_shifts_by_assignment(
                viable_shifts,
                target_h0,
                r,
                pos_plus,
                c[0]
            );

        if (shifts1.empty()) {
            continue;
        }

        std::vector<int> shifts2 =
            filter_shifts_by_assignment(
                shifts1,
                target_h0,
                r,
                pos_minus,
                c[1]
            );

        if (shifts2.empty()) {
            continue;
        }

        ok = ok && apply_value(f, pos_plus, c[0], weight, changed, supp_f);
        ok = ok && apply_value(f, pos_minus, c[1], weight, changed, supp_f);

        bool found_here = false;

        if (ok) {
            found_here = dfs_find_target_shift_pruned(
                order,
                suffix_max_add,
                idx + 1,
                r,
                target_weight,
                f,
                weight,
                target_h0,
                supp_f,
                shifts2,
                result,
                opt
            );
        }

        rollback(f, weight, changed, supp_f);

        if (found_here) {
            return true;
        }
    }
    return false;
}

SearchResult recover_one_instance(
    const std::vector<int>& D,
    const std::vector<uint8_t>& target_h0,
    int r,
    int w_half,
    const SearchOptions& opt
) {
    SearchResult result;

    if (static_cast<int>(target_h0.size()) != r) {
        throw std::runtime_error("target_h0.size() != r");
    }

    if (hamming_weight(target_h0) != w_half) {
        return result;
    }

    std::vector<int> supp_h0 = support_of_binary_vector(target_h0);

    if (static_cast<int>(supp_h0.size()) != w_half) {
        return result;
    }

    std::vector<uint8_t> f(static_cast<std::size_t>(r), 0);
    std::vector<int> supp_f;
    int weight = 0;

    auto set_one_initial = [&](int pos) {
        pos = mod_pos(pos, r);

        if (f[static_cast<std::size_t>(pos)] == 0) {
            f[static_cast<std::size_t>(pos)] = 1;
            supp_f.push_back(pos);
            ++weight;
        }
    };

    // Step 2
    set_one_initial(0);

    // Step 3: f[i] = f[-i] = 1, i=1,...,14
    for (int i = 1; i <= 14; ++i) {
        set_one_initial(i);
        set_one_initial(-i);
    }

    if (weight > w_half) {
        return result;
    }

    std::vector<int> order = build_order(D, r);

    std::vector<int> suffix_max_add(order.size() + 1, 0);

    for (int i = static_cast<int>(order.size()) - 1; i >= 0; --i) {
        int d = order[static_cast<std::size_t>(i)];
        int pos_plus = mod_pos(d, r);
        int pos_minus = mod_pos(-d, r);

        int max_add = (pos_plus == pos_minus ? 1 : 2);

        suffix_max_add[static_cast<std::size_t>(i)] =
            suffix_max_add[static_cast<std::size_t>(i + 1)] + max_add;
    }

    std::vector<int> viable_shifts =
        initial_viable_shifts(f, target_h0, r);

    if (viable_shifts.empty()) {
        return result;
    }

    dfs_find_target_shift_pruned(
        order,
        suffix_max_add,
        0,
        r,
        w_half,
        f,
        weight,
        target_h0,
        supp_f,
        viable_shifts,
        result,
        opt
    );

    return result;
}


// =============================================================================
int main(int argc, char** argv) {
    try {
        if (argc < 4) {
            std::cerr
                << "Usage:\n"
                << "  " << argv[0]
                << " input.csv r w_half [first_line_is_D=1] [max_nodes=0] [max_candidate_checks=0]\n\n"
                << "Example:\n"
                << "  " << argv[0] << " data.csv 12323 71 1 0 0\n";
            return 1;
        }

        std::string csv_file = argv[1];
        int r = std::stoi(argv[2]);
        int w_half = std::stoi(argv[3]);

        // Default: rows 1, 3, 5, ... are h0; rows 2, 4, 6, ... are D
        bool first_line_is_D = false;
        if (argc >= 5) {
            first_line_is_D = (std::stoi(argv[4]) != 0);
        }

        SearchOptions opt;

        if (argc >= 6) {
            opt.max_nodes = static_cast<std::uint64_t>(std::stoull(argv[5]));
        }

        if (argc >= 7) {
            opt.max_candidate_checks =
                static_cast<std::uint64_t>(std::stoull(argv[6]));
        }

        std::vector<Instance> instances =
            load_instances_from_csv(csv_file, r, first_line_is_D);

        std::uint64_t succeed = 0;
        std::uint64_t total_nodes = 0;
        std::uint64_t total_candidate_checks = 0;
        std::uint64_t stopped_count = 0;

        for (std::size_t i = 0; i < instances.size(); ++i) {
            const Instance& inst = instances[i];

            SearchResult res = recover_one_instance(
                inst.D,
                inst.h0,
                r,
                w_half,
                opt
            );

            if (res.found) {
                ++succeed;
            }

            if (res.stopped_by_node_limit || res.stopped_by_candidate_limit) {
                ++stopped_count;
            }

            total_nodes += res.visited_nodes;
            total_candidate_checks += res.checked_candidates;

            std::cout
                << "pair_id = " << i
                << ", found = " << (res.found ? 1 : 0)
                << ", found_shift = " << res.found_shift
                << ", visited_nodes = " << res.visited_nodes
                << ", checked_candidates = " << res.checked_candidates;

            if (res.stopped_by_node_limit) {
                std::cout << ", stopped_by_node_limit = 1";
            }

            if (res.stopped_by_candidate_limit) {
                std::cout << ", stopped_by_candidate_limit = 1";
            }

            std::cout << "\n";
        }

        double success_rate = 0.0;

        if (!instances.empty()) {
            success_rate =
                static_cast<double>(succeed) /
                static_cast<double>(instances.size());
        }

        std::cout << "\n";
        std::cout << "total_pairs = " << instances.size() << "\n";
        std::cout << "succeed = " << succeed << "\n";
        std::cout << "success_rate = " << success_rate << "\n";
        std::cout << "total_visited_nodes = " << total_nodes << "\n";
        std::cout << "total_checked_candidates = " << total_candidate_checks << "\n";
        std::cout << "stopped_count = " << stopped_count << "\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}