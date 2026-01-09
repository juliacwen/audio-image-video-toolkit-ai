/**
 * @file rms_log_analyze.cpp
 * @brief Utility to analyze RMS (Root Mean Square) log files for audio processing
 * @author Julia Wen (wendigilane@gmail.com)
 * 
 * @details
 * This tool analyzes RMS log files to identify frames with significant differences
 * between input and output RMS values, which may indicate processing artifacts,
 * clipping, or other audio issues.
 * 
 * Example rms_log.txt format:
 * frame in_rms out_rms processed
 * 1 0.0318917 0 1
 * 2 0.00171896 1.7482e-09 1
 * 3 0.000277487 0.0319572 1
 * 4 0.000242267 0.00191502 1
 * 5 0.000206685 4.33841e-05 1
 * 
 * @note Requires C++17 or later (uses structured bindings)
 * Compile with: 
 * g++ -std=c++17 rms_log_analyze.cpp -o rms_analyze
 * g++ -std=c++17 -O2 -Wall -Wextra src/rms_log_analyze.cpp -o rms_log_analyze
 * 
 * @par Revision History
 * - 01-07-2026 — Initial Checkin
 * - 01-08-2026 — Updated to output sorted csv
 */

#include "../inc/rms_analyze.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <queue>
#include <iomanip>

// ------------------- Constructor -------------------
// Reads RMS log data from file
// Time Complexity: O(N) where N = number of frames in file
// Space Complexity: O(N) to store all frame data
RMSAnalyzer::RMSAnalyzer(const std::string& filename) {
    load_file(filename);
}

// ------------------- Load File -------------------
// Parses file with format: frame in_rms out_rms processed
// Skips header line if present and handles malformed lines gracefully
// Time Complexity: O(N) where N = number of lines in file
// Space Complexity: O(N) for storing valid frame data
void RMSAnalyzer::load_file(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "ERROR: Cannot open file: " << filename << "\n";
        exit(EXIT_FAILURE); // fail fast on I/O errors
    }
    
    data_.clear();
    std::string line;
    bool skipped_header = false;
    size_t line_num = 0;
    
    while (std::getline(infile, line)) {
        line_num++;
        if (line.empty()) continue;
        
        // Skip header line if it contains "frame" keyword
        if (!skipped_header && line.find("frame") != std::string::npos) {
            skipped_header = true;
            continue;
        }
        
        std::istringstream ss(line);
        FrameRMS f;
        ss >> f.frame >> f.in_rms >> f.out_rms >> f.processed;
        
        if (!ss) {
            std::cerr << "Warning: Skipping malformed line " << line_num << ": " << line << "\n";
            continue;
        }
        
        data_.push_back(f);
    }
    
    if (data_.empty()) {
        std::cerr << "ERROR: No valid data found in file: " << filename << "\n";
        exit(EXIT_FAILURE);
    }
    
    std::cout << "Loaded " << data_.size() << " frames from " << filename << "\n";
}

// ------------------- Global Max -------------------
// Finds frame with maximum |out_rms - in_rms| across entire dataset
// This is useful for finding the single worst frame in the entire recording
// Time Complexity: O(N), requires scanning all frames once (optimal)
// Space Complexity: O(1), only stores current maximum
// Returns: tuple of <max_diff, frame_number, in_rms, out_rms>
std::tuple<double,int,double,double> RMSAnalyzer::find_global_max() const {
    if (data_.empty()) {
        std::cerr << "ERROR: No data available for analysis\n";
        return {0.0, -1, 0.0, 0.0};
    }
    
    double max_diff = -1.0;
    int max_frame = -1;
    double in_val = 0, out_val = 0;
    
    for (const auto& f : data_) {
        double diff = std::abs(f.out_rms - f.in_rms);
        if (diff > max_diff) {
            max_diff = diff;
            max_frame = f.frame;
            in_val = f.in_rms;
            out_val = f.out_rms;
        }
    }
    
    return {max_diff, max_frame, in_val, out_val};
}

// ------------------- Window Max -------------------
// Computes maximum difference per window of size window_size
// This is useful for tracking worst frame in each time segment (e.g., per second)
// Time Complexity: O(N), processes each frame exactly once
// Space Complexity: O(N/window_size) for result vector
// Returns: vector of tuples <frame_number, in_rms, out_rms> for each window's max
std::vector<std::tuple<int,double,double>> RMSAnalyzer::find_window_max(size_t window_size) const {
    if (data_.empty()) {
        std::cerr << "ERROR: No data available for analysis\n";
        return {};
    }
    
    if (window_size == 0) {
        std::cerr << "ERROR: window_size must be greater than 0\n";
        return {};
    }
    
    std::vector<std::tuple<int,double,double>> result;
    result.reserve((data_.size() + window_size - 1) / window_size);
    
    for (size_t i = 0; i < data_.size(); i += window_size) {
        double max_diff = -1.0;
        int max_frame = -1;
        double in_val = 0, out_val = 0;
        
        size_t window_end = std::min(i + window_size, data_.size());
        for (size_t j = i; j < window_end; ++j) {
            double diff = std::abs(data_[j].out_rms - data_[j].in_rms);
            if (diff > max_diff) {
                max_diff = diff;
                max_frame = data_[j].frame;
                in_val = data_[j].in_rms;
                out_val = data_[j].out_rms;
            }
        }
        
        result.emplace_back(max_frame, in_val, out_val);
    }
    
    return result;
}

// ------------------- Top K Worst -------------------
// Returns top K frames with largest difference, sorted in descending order
// This is useful for finding the K most problematic frames for manual review
// 
// Algorithm: Min-heap approach for efficiency
// - Maintains a heap of size K containing the K largest differences seen so far
// - For each frame: if diff > heap_min, remove min and insert new frame
// - Finally, extract all elements and sort descending
// 
// Time Complexity: O(N log K) where N = total frames, K = requested top frames
//   - O(N) iterations over all frames
//   - O(log K) per heap operation (push/pop)
//   - O(K log K) final sort (negligible if K << N)
//   - Much better than O(N log N) full sort when K << N (e.g., K=100, N=1M)
// 
// Space Complexity: O(K) for heap storage
// 
// Returns: vector of tuples <diff, frame_number, in_rms, out_rms> in descending order by diff
std::vector<std::tuple<int,double,double,double>> RMSAnalyzer::find_top_k(size_t K) const {
    if (data_.empty()) {
        std::cerr << "ERROR: No data available for analysis\n";
        return {};
    }
    
    if (K == 0) {
        std::cerr << "ERROR: K must be greater than 0\n";
        return {};
    }
    
    // Use min-heap to efficiently track top K largest elements
    // Min-heap keeps smallest of the "top K" at root for easy comparison
    using DiffTuple = std::tuple<double,int,double,double>;
    auto cmp = [](const DiffTuple& a, const DiffTuple& b) {
        return std::get<0>(a) > std::get<0>(b);  // Min-heap: smallest diff at top
    };
    std::priority_queue<DiffTuple, std::vector<DiffTuple>, decltype(cmp)> min_heap(cmp);
    
    for (const auto& f : data_) {
        double diff = std::abs(f.out_rms - f.in_rms);
        
        if (min_heap.size() < K) {
            // Heap not full yet, just add
            min_heap.push({diff, f.frame, f.in_rms, f.out_rms});
        } else if (diff > std::get<0>(min_heap.top())) {
            // Found larger diff than current minimum in top K
            min_heap.pop();
            min_heap.push({diff, f.frame, f.in_rms, f.out_rms});
        }
    }
    
    // Extract results from heap and sort descending
    std::vector<std::tuple<int,double,double,double>> result;
    result.reserve(min_heap.size());
    while (!min_heap.empty()) {
        auto [diff, frame, in_rms, out_rms] = min_heap.top();
        min_heap.pop();
        result.emplace_back(frame, diff, in_rms, out_rms);  // Reorder to match return type
    }
    
    // Reverse to get descending order (heap gave us ascending)
    std::reverse(result.begin(), result.end());
    
    return result;
}

// ------------------- Full Sort -------------------
// Sorts all frames by difference in descending order
// Use this when you need complete sorted results (e.g., for percentile analysis)
// Time Complexity: O(N log N), comparison-based sort is optimal for this problem
// Space Complexity: O(N) for storing all frames with computed differences
// Returns: vector of tuples <diff, frame_number, in_rms, out_rms> sorted by diff descending
std::vector<std::tuple<int,double,double,double>> RMSAnalyzer::sort_all() const {
    if (data_.empty()) {
        std::cerr << "ERROR: No data available for analysis\n";
        return {};
    }
    
    std::vector<std::tuple<int,double,double,double>> diffs;
    diffs.reserve(data_.size());
    
    for (const auto& f : data_) {
        double diff = std::abs(f.out_rms - f.in_rms);
        diffs.emplace_back(f.frame, diff, f.in_rms, f.out_rms);
    }
    
    std::sort(diffs.begin(), diffs.end(),
              [](const auto& a, const auto& b) { return std::get<1>(a) > std::get<1>(b); });
    
    return diffs;
}

// ------------------- Main Function -------------------
// Command-line interface for RMS log analysis
// Supports multiple analysis modes via command-line options:
//   --file <filename>  : specify input file (default: rms_log.txt)
//   --global           : show global max difference frame
//   --window <size>    : show max per window of specified size
//   --topk <K>         : show top K worst frames
//   --sort-all         : sort and display all frames by difference, save to rms_log_sorted.csv
//   --help, -h         : display usage information
// 
// Default behavior (no options): displays global max
// Multiple options can be combined in a single run for comprehensive analysis
int main(int argc, char* argv[]) {
    std::string filename = "rms_log.txt";
    bool do_global = false;
    bool do_window = false;
    size_t window_size = 100;
    bool do_topk = false;
    size_t K = 100;
    bool do_sort_all = false;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--file" && i+1 < argc) {
            filename = argv[++i];
        } else if (arg == "--global") {
            do_global = true;
        } else if (arg == "--window" && i+1 < argc) {
            do_window = true;
            window_size = std::stoull(argv[++i]);
        } else if (arg == "--topk" && i+1 < argc) {
            do_topk = true;
            K = std::stoull(argv[++i]);
        } else if (arg == "--sort-all") {
            do_sort_all = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "RMS Log Analyzer - Identify frames with significant RMS differences\n\n"
                      << "Usage: " << argv[0] << " [options]\n\n"
                      << "Options:\n"
                      << "  --file <filename>   Input RMS log file (default: rms_log.txt)\n"
                      << "  --global            Display frame with maximum global difference\n"
                      << "  --window <size>     Display max difference per window of <size> frames\n"
                      << "  --topk <K>          Display top K frames with largest differences\n"
                      << "  --sort-all          Sort all frames and save to rms_log_sorted.csv\n"
                      << "  --help, -h          Show this help message\n\n"
                      << "Default behavior: If no analysis options are specified, displays global max\n\n"
                      << "Examples:\n"
                      << "  " << argv[0] << "                              # Show global max\n"
                      << "  " << argv[0] << " --file audio.log --global --topk 10\n"
                      << "  " << argv[0] << " --window 1000 --topk 50\n"
                      << "  " << argv[0] << " --sort-all                   # Save sorted results to file\n";
            return 0;
        } else {
            std::cerr << "ERROR: Unknown argument: " << arg << "\n";
            std::cerr << "Use --help for usage information\n";
            return 1;
        }
    }

    // If no analysis options specified, default to global max
    if (!do_global && !do_window && !do_topk && !do_sort_all) {
        do_global = true;
        std::cout << "No analysis option specified, defaulting to --global\n\n";
    }

    // Load and analyze data
    RMSAnalyzer analyzer(filename);
    std::cout << std::fixed << std::setprecision(6);  // Format floating point output

    if (do_global) {
        auto [diff, frame, in_val, out_val] = analyzer.find_global_max();
        std::cout << "\n=== Global Max Difference ===\n"
                  << "  Difference: " << diff << "\n"
                  << "  Frame:      " << frame << "\n"
                  << "  in_rms:     " << in_val << "\n"
                  << "  out_rms:    " << out_val << "\n";
    }

    if (do_window) {
        auto res = analyzer.find_window_max(window_size);
        std::cout << "\n=== Window Max (window_size=" << window_size << ") ===\n";
        for (size_t i = 0; i < res.size(); ++i) {
            auto [f, in_val, out_val] = res[i];
            double diff = std::abs(out_val - in_val);
            std::cout << "  Window " << i << ": Frame=" << f 
                      << " Diff=" << diff
                      << " in_rms=" << in_val 
                      << " out_rms=" << out_val << "\n";
        }
    }

    if (do_topk) {
        auto res = analyzer.find_top_k(K);
        std::cout << "\n=== Top " << K << " Worst Frames ===\n";
        for (size_t i = 0; i < res.size(); ++i) {
            auto [f, diff, in_val, out_val] = res[i];
            std::cout << "  " << (i+1) << ". Diff=" << diff 
                      << " Frame=" << f
                      << " in_rms=" << in_val 
                      << " out_rms=" << out_val << "\n";
        }
    }

    if (do_sort_all) {
        auto res = analyzer.sort_all();
        
        // Write to CSV file
        std::string output_file = "rms_log_sorted.csv";
        std::ofstream outfile(output_file);
        if (!outfile) {
            std::cerr << "ERROR: Cannot create output file: " << output_file << "\n";
            return 1;
        }
        
        // Write CSV header
        outfile << std::fixed << std::setprecision(6);
        outfile << "frame,diff,in_rms,out_rms\n";
        
        // Write all sorted results
        for (const auto& [f, diff, in_val, out_val] : res) {
            outfile << f << "," << diff << "," << in_val << "," << out_val << "\n";
        }
        
        outfile.close();
        std::cout << "\n=== All Frames Sorted by Difference ===\n";
        std::cout << "Results saved to: " << output_file << " (CSV format)\n";
        std::cout << "Total frames: " << res.size() << "\n";
        
        // Display top 10 as preview
        std::cout << "\nTop 10 preview:\n";
        size_t preview_count = std::min(size_t(10), res.size());
        for (size_t i = 0; i < preview_count; ++i) {
            auto [f, diff, in_val, out_val] = res[i];
            std::cout << "  " << (i+1) << ". Diff=" << diff 
                      << " Frame=" << f
                      << " in_rms=" << in_val 
                      << " out_rms=" << out_val << "\n";
        }
    }

    return 0;
}