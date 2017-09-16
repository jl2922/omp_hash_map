#ifndef OMP_HASH_MAP_H_
#define OMP_HASH_MAP_H_

#include <array>
#include <functional>
#include <memory>
#include <vector>
#include "omp.h"

namespace cornell {
// A high performance concurrent hash map based on OpenMP.
template <class K, class V, class H = std::hash<K>>
class omp_hash_map {
 public:
  omp_hash_map();

  ~omp_hash_map();

  // Set the number of buckets in the container to contain at least n buckets.
  void reserve(const size_t n_buckets);

  // Return the number of buckets.
  size_t get_n_buckets() const { return n_buckets; };

  // Return the current load factor (the ratio between the number of keys and buckets).
  double get_load_factor() const { return 1.0 * n_keys / n_buckets; }

  // Return the number of keys.
  size_t get_n_keys() const { return n_keys; }

  // Set the specified key to the specified value.
  void set(const K& key, const V& value);

  // Update the value of the specified key.
  // If the key does not exist, construct it with the default initializer first.
  void set(const K& key, const std::function<void(V&)>& setter);

  // Update the value of the specified key.
  // If the key does not exist, construct and set it to the default value first.
  void set(const K& key, const std::function<void(V&)>& setter, const V& default_value);

  // Remove the key.
  void unset(const K& key);

  // Test if the specified key exists.
  bool has(const K& key);

  // Return the mapped value for the value of the specific key.
  // If the key does not exist, return the default value.
  template <class W>
  W map(const K& key, const std::function<W(const V&)>& mapper, const W& default_value);

  // Return the reduced value of the mapped values of all the keys.
  // If no key exists, return the default value.
  template <class W>
  W map_reduce(
      const std::function<W(const K&, const V&)>& mapper,
      const std::function<void(W&, const W&)>& reducer,
      const W& default_value);

  // Apply the handler to the value of the specific key, if it exists.
  void apply(const K& key, const std::function<void(const V&)>& handler);

  // Apply the handler to all the keys.
  void apply(const std::function<void(const K&, const V&)>& handler);

  // Clear all keys.
  void clear();

 private:
  size_t n_keys;

  size_t n_buckets;

  size_t n_threads;

  size_t n_segments;

  H hasher;

  std::vector<omp_lock_t> segment_locks;

  std::vector<omp_lock_t> rehashing_segment_locks;

  constexpr static size_t N_INITIAL_BUCKETS = 3;

  constexpr static size_t N_SEGMENTS_PER_THREAD = 7;

  struct hash_node {
    K key;
    V value;
    std::unique_ptr<hash_node> next;
    hash_node(const K& key, const V& value) : key(key), value(value){};
  };

  std::vector<std::unique_ptr<hash_node>> buckets;

  // Get the actual number of hash buckets that is larger than or equal to the specified number.
  size_t get_n_rehashing_buckets(const size_t n_buckets);

  // Apply node_handler to the hash node which has the specific key.
  // If the key does not exist, apply to the unassociated node from the corresponding bucket.
  void hash_node_apply(
      const K& key, const std::function<void(std::unique_ptr<hash_node>&)>& node_handler);

  // Apply node_handler to all the hash nodes.
  void hash_node_apply(const std::function<void(std::unique_ptr<hash_node>&)>& node_handler);

  // Recursive part of hash_node_apply with a specific key.
  void hash_node_apply_recursive(
      std::unique_ptr<hash_node>& node,
      const K& key,
      const std::function<void(std::unique_ptr<hash_node>&)>& node_handler);

  // Recursive part of hash_node_apply to all the keys.
  void hash_node_apply_recursive(
      std::unique_ptr<hash_node>& node,
      const std::function<void(std::unique_ptr<hash_node>&)>& node_handler);

  void lock_all_segments();

  void unlock_all_segments();
};

template <class K, class V, class H>
omp_hash_map<K, V, H>::omp_hash_map() {
  n_keys = 0;
  n_buckets = N_INITIAL_BUCKETS;
  n_threads = omp_get_max_threads();
  n_segments = n_threads * N_SEGMENTS_PER_THREAD;

  segment_locks.resize(n_segments);
  rehashing_segment_locks.resize(n_segments);
  for (auto& lock : segment_locks) omp_init_lock(&lock);
  for (auto& lock : rehashing_segment_locks) omp_init_lock(&lock);

  buckets.resize(n_buckets);
}

template <class K, class V, class H>
omp_hash_map<K, V, H>::~omp_hash_map() {
  for (auto& lock : segment_locks) omp_destroy_lock(&lock);
  for (auto& lock : rehashing_segment_locks) omp_destroy_lock(&lock);
}

template <class K, class V, class H>
void omp_hash_map<K, V, H>::reserve(const size_t n_buckets_in) {
  lock_all_segments();
  if (n_buckets >= n_buckets_in) {
    unlock_all_segments();
    return;
  }

  const size_t n_rehashing_buckets = get_n_rehashing_buckets(n_buckets_in);

  // Rehash.
  std::vector<std::unique_ptr<hash_node>> rehashing_buckets(n_rehashing_buckets);
  const auto& node_handler = [&](std::unique_ptr<hash_node>& node) {
    const K& key = node->key;
    const size_t hash_value = hasher(key);
    const size_t bucket_id = hash_value % n_rehashing_buckets;
    const auto& rehashing_node_handler = [&](std::unique_ptr<hash_node>& rehashing_node) {
      rehashing_node = std::move(node);
      rehashing_node->next.release();
    };
    const size_t segment_id = hash_value % n_segments;
    auto& lock = rehashing_segment_locks[segment_id];
    omp_set_lock(&lock);
    hash_node_apply_recursive(rehashing_buckets[bucket_id], key, rehashing_node_handler);
    omp_unset_lock(&lock);
  };

#pragma omp parallel for schedule(static, 1)
  for (size_t i = 0; i < n_buckets; i++) {
    hash_node_apply_recursive(buckets[i], node_handler);
  }

  buckets = std::move(rehashing_buckets);
  n_buckets = n_rehashing_buckets;
  unlock_all_segments();
}

template <class K, class V, class H>
size_t omp_hash_map<K, V, H>::get_n_rehashing_buckets(const size_t n_buckets) {
  constexpr size_t PRIME_NUMBERS[] = {
      5,         11,        23,        47,         97,        199,      409,      823,
      1741,      3469,      6949,      14033,      28411,     57557,    116731,   236897,
      480881,    976369,    1982627,   4026031,    8175383,   16601593, 33712729, 68460391,
      139022417, 282312799, 573292817, 1164186217, 2147483647};
  constexpr size_t N_PRIME_NUMBERS = sizeof(PRIME_NUMBERS) / sizeof(size_t);
  constexpr size_t LAST_PRIME_NUMBER = PRIME_NUMBERS[N_PRIME_NUMBERS - 1];
  size_t remaining_factor = n_buckets;
  size_t n_rehashing_buckets = 1;
  if (remaining_factor > LAST_PRIME_NUMBER) {
    remaining_factor /= 817504253;
    n_rehashing_buckets *= 817504253;
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

template <class K, class V, class H>
void omp_hash_map<K, V, H>::set(const K& key, const V& value) {
  const auto& node_handler = [&](std::unique_ptr<hash_node>& node) {
    if (!node) {
      node.reset(new hash_node(key, value));
#pragma omp atomic
      n_keys++;
    } else {
      node->value = value;
    }
  };
  if (n_keys >= n_buckets) reserve(n_keys);
  hash_node_apply(key, node_handler);
}

template <class K, class V, class H>
void omp_hash_map<K, V, H>::set(const K& key, const std::function<void(V&)>& setter) {
  const auto& node_handler = [&](std::unique_ptr<hash_node>& node) {
    if (!node) {
      V value;
      setter(value);
      node.reset(new hash_node(key, value));
#pragma omp atomic
      n_keys++;
    } else {
      setter(node->value);
    }
  };
  if (n_keys >= n_buckets) reserve(n_keys);
  hash_node_apply(key, node_handler);
}

template <class K, class V, class H>
void omp_hash_map<K, V, H>::set(
    const K& key, const std::function<void(V&)>& setter, const V& default_value) {
  const auto& node_handler = [&](std::unique_ptr<hash_node>& node) {
    if (!node) {
      V value(default_value);
      setter(value);
      node.reset(new hash_node(key, value));
#pragma omp atomic
      n_keys++;
    } else {
      setter(node->value);
    }
  };
  if (n_keys >= n_buckets) reserve(n_keys);
  hash_node_apply(key, node_handler);
}

template <class K, class V, class H>
void omp_hash_map<K, V, H>::unset(const K& key) {
  const auto& node_handler = [&](std::unique_ptr<hash_node>& node) {
    if (node) {
      node = std::move(node->next);
#pragma omp atomic
      n_keys--;
    }
  };
  hash_node_apply(key, node_handler);
}

template <class K, class V, class H>
bool omp_hash_map<K, V, H>::has(const K& key) {
  bool has_key = false;
  const auto& node_handler = [&](std::unique_ptr<hash_node>& node) {
    if (node) has_key = true;
  };
  hash_node_apply(key, node_handler);
  return has_key;
}

template <class K, class V, class H>
template <class W>
W omp_hash_map<K, V, H>::map_reduce(
    const std::function<W(const K&, const V&)>& mapper,
    const std::function<void(W&, const W&)>& reducer,
    const W& default_value) {
  std::vector<W> mapped_values(n_threads, default_value);
  W reduced_value = default_value;
  const auto& node_handler = [&](std::unique_ptr<hash_node>& node) {
    const size_t thread_id = omp_get_thread_num();
    const W& mapped_value = mapper(node->key, node->value);
    fprintf(stderr, "%d\n", mapped_value);
    reducer(mapped_values[thread_id], mapped_value);
  };
  hash_node_apply(node_handler);
  for (const auto& mapped_value : mapped_values) reducer(reduced_value, mapped_value);
  return reduced_value;
}

template <class K, class V, class H>
void omp_hash_map<K, V, H>::apply(const K& key, const std::function<void(const V&)>& handler) {
  const auto& node_handler = [&](std::unique_ptr<hash_node>& node) {
    if (node) handler(node->value);
  };
  hash_node_apply(key, node_handler);
}

template <class K, class V, class H>
void omp_hash_map<K, V, H>::clear() {
  lock_all_segments();
  buckets.resize(N_INITIAL_BUCKETS);
  for (auto& bucket : buckets) bucket.reset();
  n_keys = 0;
  unlock_all_segments();
}

template <class K, class V, class H>
void omp_hash_map<K, V, H>::hash_node_apply(
    const K& key, const std::function<void(std::unique_ptr<hash_node>&)>& node_handler) {
  const size_t hash_value = hasher(key);
  const size_t segment_id = hash_value % n_segments;
  auto& lock = segment_locks[segment_id];
  omp_set_lock(&lock);
  const size_t bucket_id = hash_value % n_buckets;
  hash_node_apply_recursive(buckets[bucket_id], key, node_handler);
  omp_unset_lock(&lock);
}

template <class K, class V, class H>
void omp_hash_map<K, V, H>::hash_node_apply(
    const std::function<void(std::unique_ptr<hash_node>&)>& node_handler) {
  lock_all_segments();
// For a good hash function, a static schedule shall provide both balance and speed.
#pragma omp parallel for schedule(static, 1)
  for (size_t i = 0; i < n_buckets; i++) {
    hash_node_apply_recursive(buckets[i], node_handler);
  }
  unlock_all_segments();
}

template <class K, class V, class H>
void omp_hash_map<K, V, H>::hash_node_apply_recursive(
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

template <class K, class V, class H>
void omp_hash_map<K, V, H>::hash_node_apply_recursive(
    std::unique_ptr<hash_node>& node,
    const std::function<void(std::unique_ptr<hash_node>&)>& node_handler) {
  if (node) {
    hash_node_apply_recursive(node->next, node_handler);
    node_handler(node);
  }
}

template <class K, class V, class H>
void omp_hash_map<K, V, H>::lock_all_segments() {
  for (auto& lock : segment_locks) omp_set_lock(&lock);
}

template <class K, class V, class H>
void omp_hash_map<K, V, H>::unlock_all_segments() {
  for (auto& lock : segment_locks) omp_unset_lock(&lock);
}
};
#endif