#ifndef OMP_HASH_MAP_H_
#define OMP_HASH_MAP_H_

#include <functional>
#include <vector>
#include "omp.h"

namespace cornell {
template <class K, class V, class H = std::hash<K>>
class omp_hash_map {
 public:
  omp_hash_map();

  ~omp_hash_map();

  void reserve(const size_t n_buckets);

  size_t bucket_count() const { return n_buckets; };

  size_t size() const { return n_keys; }

  double hash_load() const { return 1.0 * n_keys / n_buckets; }

 private:
  size_t n_keys;

  size_t n_buckets;

  size_t n_threads;

  size_t n_segments;

  std::vector<omp_lock_t> segment_locks;

  omp_lock_t n_keys_lock;

  const size_t N_INITIAL_BUCKETS = 5;

  const size_t N_SEGMENTS_PER_THREAD = 7;
};

template <class K, class V, class H>
omp_hash_map<K, V, H>::omp_hash_map() {
  n_keys = 0;
  n_buckets = N_INITIAL_BUCKETS;
  n_threads = omp_get_max_threads();

  // Initialize locks.
  n_segments = n_threads * N_SEGMENTS_PER_THREAD;
  segment_locks.resize(n_segments);
  for (auto& segment_lock : segment_locks) omp_init_lock(&segment_lock);
  omp_init_lock(&n_keys_lock);
}

template <class K, class V, class H>
omp_hash_map<K, V, H>::~omp_hash_map() {
  for (auto& segment_lock : segment_locks) omp_destroy_lock(&segment_lock);
  omp_destroy_lock(&n_keys_lock);
}
}
#endif