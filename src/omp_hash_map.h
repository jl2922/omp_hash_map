#ifndef OMP_HASH_MAP_H_
#define OMP_HASH_MAP_H_

#include <functional>

namespace cornell {
template <class K, class V, class H = std::hash<K>>
class omp_hash_map {
 public:
  omp_hash_map();

  size_t size() { return n_keys; }

 private:
  size_t n_keys;
};

template <class K, class V, class H>
omp_hash_map<K, V, H>::omp_hash_map() {
  n_keys = 0;
}
}
#endif