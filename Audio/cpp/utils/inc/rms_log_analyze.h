/**
 * @file rms_log_analyze.h
 * @brief Header for RMS (Root Mean Square) log analysis utility
 * @author Julia Wen (wendigilane@gmail.com)
 * 
 * @details
 * Provides functionality to analyze audio processing logs containing RMS values.
 * Useful for identifying frames with processing artifacts, clipping, or anomalies
 * by comparing input vs output RMS values.
 * 
 * @note Requires C++17 or later
 * 
 * @par Revision History
 * - 01-07-2026 — Initial Checkin
 */

#ifndef RMS_LOG_ANALYZE_H
#define RMS_LOG_ANALYZE_H

#include <vector>
#include <tuple>
#include <string>

// Struct representing one RMS frame record from the log file
struct FrameRMS {
    int frame;      // Frame number (sequential identifier)
    double in_rms;  // Input RMS value (root mean square of input signal)
    double out_rms; // Output RMS value (root mean square after processing)
    bool processed; // Processing status flag (1 = processed, 0 = skipped)
};

/**
 * @class RMSAnalyzer
 * @brief Analyzes RMS log data to identify frames with anomalous differences
 * 
 * This class loads RMS log files and provides various analysis methods to find
 * frames where |out_rms - in_rms| is large, which may indicate audio issues.
 */
class RMSAnalyzer {
public:
    /**
     * @brief Constructor - loads and parses RMS log file
     * @param filename Path to RMS log file
     * @throws Exits program if file cannot be opened or contains no valid data
     * 
     * Time Complexity: O(N) where N = number of lines in file
     * Space Complexity: O(N) to store frame data
     */
    explicit RMSAnalyzer(const std::string& filename);

    /**
     * @brief Find frame with maximum |out_rms - in_rms| across entire dataset
     * @return Tuple of <max_diff, frame_number, in_rms, out_rms>
     * 
     * Time Complexity: O(N) - single pass, optimal for this problem
     * Space Complexity: O(1) - constant extra space
     */
    std::tuple<double,int,double,double> find_global_max() const;

    /**
     * @brief Find maximum difference frame within each window
     * @param window_size Number of frames per window
     * @return Vector of tuples <frame_number, in_rms, out_rms>, one per window
     * 
     * Useful for tracking worst frame in each time segment (e.g., per second of audio)
     * 
     * Time Complexity: O(N) - each frame examined exactly once
     * Space Complexity: O(N/window_size) - one result per window
     */
    std::vector<std::tuple<int,double,double>> find_window_max(size_t window_size) const;

    /**
     * @brief Find top K frames with largest differences, sorted descending
     * @param K Number of top frames to return
     * @return Vector of tuples <frame_number, diff, in_rms, out_rms> sorted by diff
     * 
     * Uses min-heap algorithm for efficiency when K << N
     * 
     * Time Complexity: O(N log K) where N = total frames, K = requested results
     *   - Significantly better than O(N log N) full sort when K is small
     *   - Example: K=100, N=1,000,000 → ~10x faster than full sort
     * 
     * Space Complexity: O(K) - heap storage only
     */
    std::vector<std::tuple<int,double,double,double>> find_top_k(size_t K) const;

    /**
     * @brief Sort all frames by difference in descending order
     * @return Vector of tuples <frame_number, diff, in_rms, out_rms> fully sorted
     * 
     * Use when complete sorted results are needed (percentile analysis, full reports)
     * For finding just top K frames, use find_top_k() which is more efficient
     * 
     * Time Complexity: O(N log N) - comparison-based sort, optimal for general sorting
     * Space Complexity: O(N) - stores all frames with computed differences
     */
    std::vector<std::tuple<int,double,double,double>> sort_all() const;

private:
    std::vector<FrameRMS> data_;  ///< Storage for all loaded frame records

    /**
     * @brief Helper to parse and load RMS log file
     * @param filename Path to file
     * 
     * Handles header lines, skips malformed entries, validates data exists
     */
    void load_file(const std::string& filename);
};

#endif // RMS_LOG_ANALYZE_H