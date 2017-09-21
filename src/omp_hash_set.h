#ifndef omp_hash_set_H_
#define omp_hash_set_H_

#include <array>
#include <functional>
#include <memory>
#include <vector>
#include "omp.h"

namespace cornell {
namespace hci {
// A high performance concurrent hash map based on OpenMP.
template <class K, class H = std::hash<K>>
class omp_hash_set {
 public:
  omp_hash_set();

  ~omp_hash_set();

  // Set the number of buckets in the container to be at least the specified value.
  void reserve(const size_t n_buckets) {
    const size_t n_rehashing_buckets = get_n_rehashing_buckets(n_buckets);
    rehash(n_rehashing_buckets);
  };

  // Return the number of buckets.
  size_t get_n_buckets() const { return n_buckets; };

  // Return the current load factor (the ratio between the number of keys and buckets).
  double get_load_factor() const { return static_cast<double>(n_keys) / n_buckets; }

  // Return the max load factor beyond which an automatic rehashing will occur.
  double get_max_load_factor() const { return max_load_factor; }

  // Set the max load factor beyond which an automatic rehashing will occur.
  void set_max_load_factor(const double max_load_factor) {
    this->max_load_factor = max_load_factor;
  }

  // Return the number of keys.
  size_t get_n_keys() const { return n_keys; }

  // Set the specified key.
  void add(const K& key);

  // Remove the specified key.
  void remove(const K& key);

  // Test if the specified key exists.
  bool has(const K& key);

  // Return the reduced value of the mapped values of all the keys.
  // If no key exists, return the default value.
  template <class W>
  W map_reduce(
      const std::function<W(const K&)>& mapper,
      const std::function<void(W&, const W&)>& reducer,
      const W& default_value);

  // Apply the handler to all the keys.
  void apply(const std::function<void(const K&)>& handler);

  // Clear all keys.
  void clear();

 private:
  size_t n_keys;

  size_t n_buckets;

  double max_load_factor;

  size_t n_threads;

  // The entire hash map is divided into several segments (depends on how many threads).
  // Each segment can be locked and accessed independently in parallel.
  size_t n_segments;

  H hasher;

  std::vector<omp_lock_t> segment_locks;

  // For parallel rehashing (Require omp_set_nested(1)).
  std::vector<omp_lock_t> rehashing_segment_locks;

  constexpr static size_t N_INITIAL_BUCKETS = 11;

  constexpr static size_t N_SEGMENTS_PER_THREAD = 7;

  constexpr static double DEFAULT_MAX_LOAD_FACTOR = 1.0;

  struct hash_node {
    K key;
    std::unique_ptr<hash_node> next;
    hash_node(const K& key) : key(key){};
  };

  std::vector<std::unique_ptr<hash_node>> buckets;

  // Set the number of buckets to be at least the number of current keys times max load factor.
  void rehash() { reserve(n_keys / max_load_factor); }

  void rehash(const size_t n_rehashing_buckets);

  // Get the number of hash buckets to use.
  // This number shall be larger than or equal to the specified number.
  size_t get_n_rehashing_buckets(const size_t n_buckets) const;

  // Apply node_handler to the hash node which has the specific key.
  // If the key does not exist, apply to the unassociated node from the corresponding bucket.
  void hash_node_apply(
      const K& key, const std::function<void(std::unique_ptr<hash_node>&)>& node_handler);

  // Apply node_handler to all the hash nodes.
  void hash_node_apply(const std::function<void(std::unique_ptr<hash_node>&)>& node_handler);

  // Recursively find the node with the specified key on the list starting from the node specified.
  // Then apply the specified handler to that node.
  // If the key does not exist, apply the handler to the unassociated node at the end of the list.
  void hash_node_apply_recursive(
      std::unique_ptr<hash_node>& node,
      const K& key,
      const std::function<void(std::unique_ptr<hash_node>&)>& node_handler);

  // Recursively apply the handler to each node on the list from the node specified (post-order).
  void hash_node_apply_recursive(
      std::unique_ptr<hash_node>& node,
      const std::function<void(std::unique_ptr<hash_node>&)>& node_handler);

  void lock_all_segments();

  void unlock_all_segments();
};

template <class K, class H>
omp_hash_set<K, H>::omp_hash_set() {
  n_keys = 0;
  n_buckets = N_INITIAL_BUCKETS;
  buckets.resize(n_buckets);
  max_load_factor = DEFAULT_MAX_LOAD_FACTOR;

  n_threads = omp_get_max_threads();
  n_segments = n_threads * N_SEGMENTS_PER_THREAD;
  segment_locks.resize(n_segments);
  rehashing_segment_locks.resize(n_segments);
  for (auto& lock : segment_locks) omp_init_lock(&lock);
  for (auto& lock : rehashing_segment_locks) omp_init_lock(&lock);
}

template <class K, class H>
omp_hash_set<K, H>::~omp_hash_set() {
  for (auto& lock : segment_locks) omp_destroy_lock(&lock);
  for (auto& lock : rehashing_segment_locks) omp_destroy_lock(&lock);
}

template <class K, class H>
void omp_hash_set<K, H>::rehash(const size_t n_rehashing_buckets) {
  lock_all_segments();

  // No decrease in the number of buckets.
  if (n_buckets >= n_rehashing_buckets) {
    unlock_all_segments();
    return;
  }

  // Rehash.
  std::vector<std::unique_ptr<hash_node>> rehashing_buckets(n_rehashing_buckets);
  const auto& node_handler = [&](std::unique_ptr<hash_node>& node) {
    const auto& rehashing_node_handler = [&](std::unique_ptr<hash_node>& rehashing_node) {
      rehashing_node = std::move(node);
      rehashing_node->next.reset();
    };
    const K& key = node->key;
    const size_t hash_value = hasher(key);
    const size_t bucket_id = hash_value % n_rehashing_buckets;
    const size_t segment_id = bucket_id % n_segments;
    auto& lock = rehashing_segment_locks[segment_id];
    omp_set_lock(&lock);
    hash_node_apply_recursive(rehashing_buckets[bucket_id], key, rehashing_node_handler);
    omp_unset_lock(&lock);
  };
#pragma omp parallel for
  for (size_t i = 0; i < n_buckets; i++) {
    hash_node_apply_recursive(buckets[i], node_handler);
  }

  buckets = std::move(rehashing_buckets);
  n_buckets = n_rehashing_buckets;
  unlock_all_segments();
}

template <class K, class H>
size_t omp_hash_set<K, H>::get_n_rehashing_buckets(const size_t n_buckets_in) const {
  // Returns a number that is greater than or equal to n_buckets_in.
  // That number is either a prime number itself, or a product of two prime numbers.
  // Returns a number that is greater than or equal to n_buckets_in.
  // That number is either a prime number itself, or a product of two prime numbers.
  constexpr size_t PRIME_NUMBERS[] = {11,   17,    29,    47,    79,    127,   211,
                                      337,  547,   887,   1433,  2311,  3739,  6053,
                                      9791, 15858, 25667, 41539, 67213, 104729};
  constexpr size_t N_PRIME_NUMBERS = sizeof(PRIME_NUMBERS) / sizeof(size_t);
  constexpr size_t LAST_PRIME_NUMBER = PRIME_NUMBERS[N_PRIME_NUMBERS - 1];
  constexpr size_t DIVISION_FACTOR = 15858;
  size_t remaining_factor = n_buckets_in;
  size_t n_rehashing_buckets = 1;
  for (size_t i = 0; i < 3; i++) {
    if (remaining_factor > LAST_PRIME_NUMBER) {
      remaining_factor /= DIVISION_FACTOR;
      n_rehashing_buckets *= DIVISION_FACTOR;
    }
  }
  if (remaining_factor > LAST_PRIME_NUMBER) throw std::invalid_argument("n_buckets too large");
  size_t left = 0, right = N_PRIME_NUMBERS - 1;
  while (left < right) {
    size_t mid = (left + right) / 2;
    if (PRIME_NUMBERS[mid] < remaining_factor) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  n_rehashing_buckets *= PRIME_NUMBERS[left];
  return n_rehashing_buckets;
}

template <class K, class H>
void omp_hash_set<K, H>::add(const K& key) {
  const auto& node_handler = [&](std::unique_ptr<hash_node>& node) {
    if (!node) {
      node.reset(new hash_node(key));
#pragma omp atomic
      n_keys++;
    }
  };
  hash_node_apply(key, node_handler);
  if (n_keys >= n_buckets * max_load_factor) rehash();
}

template <class K, class H>
void omp_hash_set<K, H>::remove(const K& key) {
  const auto& node_handler = [&](std::unique_ptr<hash_node>& node) {
    if (node) {
      node = std::move(node->next);
#pragma omp atomic
      n_keys--;
    }
  };
  hash_node_apply(key, node_handler);
}

template <class K, class H>
bool omp_hash_set<K, H>::has(const K& key) {
  bool has_key = false;
  const auto& node_handler = [&](const std::unique_ptr<hash_node>& node) {
    if (node) has_key = true;
  };
  hash_node_apply(key, node_handler);
  return has_key;
}

template <class K, class H>
template <class W>
W omp_hash_set<K, H>::map_reduce(
    const std::function<W(const K&)>& mapper,
    const std::function<void(W&, const W&)>& reducer,
    const W& default_value) {
  std::vector<W> thread_reduced_values(n_threads, default_value);
  W reduced_value = default_value;
  const auto& node_handler = [&](std::unique_ptr<hash_node>& node) {
    const size_t thread_id = omp_get_thread_num();
    const W& mapped_value = mapper(node->key);
    reducer(thread_reduced_values[thread_id], mapped_value);
  };
  hash_node_apply(node_handler);
  for (const auto& value : thread_reduced_values) reducer(reduced_value, value);
  return reduced_value;
}

template <class K, class H>
void omp_hash_set<K, H>::apply(const std::function<void(const K&)>& handler) {
  const auto& node_handler = [&](std::unique_ptr<hash_node>& node) { handler(node->key); };
  hash_node_apply(node_handler);
}

template <class K, class H>
void omp_hash_set<K, H>::clear() {
  lock_all_segments();
  buckets.resize(N_INITIAL_BUCKETS);
  for (auto& bucket : buckets) bucket.reset();
  n_keys = 0;
  unlock_all_segments();
}

template <class K, class H>
void omp_hash_set<K, H>::hash_node_apply(
    const K& key, const std::function<void(std::unique_ptr<hash_node>&)>& node_handler) {
  const size_t hash_value = hasher(key);
  bool applied = false;
  while (!applied) {
    const size_t n_buckets_snapshot = n_buckets;
    const size_t bucket_id = hash_value % n_buckets_snapshot;
    const size_t segment_id = bucket_id % n_segments;
    auto& lock = segment_locks[segment_id];
    omp_set_lock(&lock);
    if (n_buckets_snapshot != n_buckets) {
      omp_unset_lock(&lock);
      continue;
    }
    hash_node_apply_recursive(buckets[bucket_id], key, node_handler);
    omp_unset_lock(&lock);
    applied = true;
  }
}

template <class K, class H>
void omp_hash_set<K, H>::hash_node_apply(
    const std::function<void(std::unique_ptr<hash_node>&)>& node_handler) {
  lock_all_segments();
// For a good hash function, a static schedule shall provide both a good balance and speed.
#pragma omp parallel for
  for (size_t i = 0; i < n_buckets; i++) {
    hash_node_apply_recursive(buckets[i], node_handler);
  }
  unlock_all_segments();
}

template <class K, class H>
void omp_hash_set<K, H>::hash_node_apply_recursive(
    std::unique_ptr<hash_node>& node,
    const K& key,
    const std::function<void(std::unique_ptr<hash_node>&)>& node_handler) {
  if (node) {
    if (node->key == key) {
      node_handler(node);
    } else {
      hash_node_apply_recursive(node->next, key, node_handler);
    }
  } else {
    node_handler(node);
  }
}

template <class K, class H>
void omp_hash_set<K, H>::hash_node_apply_recursive(
    std::unique_ptr<hash_node>& node,
    const std::function<void(std::unique_ptr<hash_node>&)>& node_handler) {
  if (node) {
    // Post-order traversal for rehashing.
    hash_node_apply_recursive(node->next, node_handler);
    node_handler(node);
  }
}

template <class K, class H>
void omp_hash_set<K, H>::lock_all_segments() {
  for (auto& lock : segment_locks) omp_set_lock(&lock);
}

template <class K, class H>
void omp_hash_set<K, H>::unlock_all_segments() {
  for (auto& lock : segment_locks) omp_unset_lock(&lock);
}
}
}
#endif