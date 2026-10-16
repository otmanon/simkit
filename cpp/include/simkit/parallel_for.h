// Minimal thread-parallel loop with no external dependency (no OpenMP).
//
// The thread count is taken from ``SIMKIT_NUM_THREADS`` if set, otherwise
// from the hardware concurrency; ``set_num_threads`` overrides it at runtime.
// Loops with fewer than ``2 * grain`` iterations run serially.
#pragma once

#include <algorithm>
#include <cstdint>
#include <exception>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace simkit {

int num_threads();
void set_num_threads(int n);

template <class Fn>
void parallel_for(std::int64_t n, Fn&& fn, std::int64_t grain = 1024) {
  if (n <= 0) return;
  std::int64_t nt = num_threads();
  if (nt <= 1 || n < 2 * grain) {
    for (std::int64_t i = 0; i < n; ++i) fn(i);
    return;
  }
  nt = std::min<std::int64_t>(nt, (n + grain - 1) / grain);
  const std::int64_t chunk = (n + nt - 1) / nt;

  std::vector<std::thread> pool;
  pool.reserve(static_cast<std::size_t>(nt));
  std::exception_ptr error;
  std::mutex error_mutex;
  for (std::int64_t k = 0; k < nt; ++k) {
    const std::int64_t begin = k * chunk;
    const std::int64_t end = std::min(n, begin + chunk);
    if (begin >= end) break;
    pool.emplace_back([&, begin, end]() {
      try {
        for (std::int64_t i = begin; i < end; ++i) fn(i);
      } catch (...) {
        std::lock_guard<std::mutex> lock(error_mutex);
        if (!error) error = std::current_exception();
      }
    });
  }
  for (auto& th : pool) th.join();
  if (error) std::rethrow_exception(error);
}

}  // namespace simkit
