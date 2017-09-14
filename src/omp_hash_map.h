#ifndef OMP_HASH_MAP_H_
#define OMP_HASH_MAP_H_

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

 private:
  size_t n_keys;

  size_t n_buckets;

  size_t n_threads;

  size_t n_segments;

  H hasher;

  std::vector<omp_lock_t> segment_locks;

  constexpr static size_t N_INITIAL_BUCKETS = 5;

  constexpr static size_t N_SEGMENTS_PER_THREAD = 7;

  struct hash_node {
    K key;
    V value;
    std::unique_ptr<hash_node> next;
    hash_node(const K& key, const V& value) : key(key), value(value){};
  };

  std::vector<std::unique_ptr<hash_node>> buckets;

  // Apply node_handler to the hash node which either has the specific key or is a null pointer.
  void hash_node_apply(
      const K& key, const std::function<void(std::unique_ptr<hash_node>&)>& node_handler);

  // Apply node_handler to the hash node which either has the specific key or is a null pointer.
  void hash_node_apply_recursive(
      std::unique_ptr<hash_node>& node,
      const K& key,
      const std::function<void(std::unique_ptr<hash_node>&)>& node_handler);
};

template <class K, class V, class H>
omp_hash_map<K, V, H>::omp_hash_map() {
  n_keys = 0;
  n_buckets = N_INITIAL_BUCKETS;
  n_threads = omp_get_max_threads();
  n_segments = n_threads * N_SEGMENTS_PER_THREAD;

  segment_locks.resize(n_segments);
  for (auto& segment_lock : segment_locks) omp_init_lock(&segment_lock);

  buckets.resize(n_buckets);
}

template <class K, class V, class H>
omp_hash_map<K, V, H>::~omp_hash_map() {
  for (auto& segment_lock : segment_locks) omp_destroy_lock(&segment_lock);
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
    if (node) {
      has_key = true;
    }
  };
  hash_node_apply(key, node_handler);
  return has_key;
}

template <class K, class V, class H>
void omp_hash_map<K, V, H>::apply(const K& key, const std::function<void(const V&)>& handler) {
  const auto& node_handler = [&](std::unique_ptr<hash_node>& node) {
    if (node) handler(node->value);
  };
  hash_node_apply(key, node_handler);
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
}
#endif