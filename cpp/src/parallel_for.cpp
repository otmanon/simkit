#include "simkit/parallel_for.h"

#include <atomic>
#include <cstdlib>
#include <string>

namespace simkit {

namespace {
std::atomic<int> g_num_threads{0};  // 0 => not yet resolved

int resolve_default() {
  if (const char* env = std::getenv("SIMKIT_NUM_THREADS")) {
    try {
      const int n = std::stoi(env);
      if (n > 0) return n;
    } catch (...) {
    }
  }
  const unsigned hw = std::thread::hardware_concurrency();
  return hw == 0 ? 1 : static_cast<int>(hw);
}
}  // namespace

int num_threads() {
  int n = g_num_threads.load();
  if (n <= 0) {
    n = resolve_default();
    g_num_threads.store(n);
  }
  return n;
}

void set_num_threads(int n) { g_num_threads.store(n <= 0 ? resolve_default() : n); }

}  // namespace simkit
