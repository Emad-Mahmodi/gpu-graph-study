// bench/timer.hpp
// Minimal high-resolution timer + a repeat-and-take-median harness.
// The study protocol (README §Reproducibility) reports the MEDIAN of N runs.
#pragma once

#include <algorithm>
#include <chrono>
#include <functional>
#include <vector>

namespace ggs {

class Timer {
public:
    void start() { t0_ = clock::now(); }
    double stop_ms() {
        auto t1 = clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0_).count();
    }
private:
    using clock = std::chrono::steady_clock;
    clock::time_point t0_;
};

// Run `fn` `repeats` times, return the median wall-clock time in ms.
inline double time_median_ms(const std::function<void()>& fn, int repeats = 10) {
    std::vector<double> samples;
    samples.reserve(repeats);
    Timer t;
    for (int i = 0; i < repeats; ++i) {
        t.start();
        fn();
        samples.push_back(t.stop_ms());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

}  // namespace ggs
