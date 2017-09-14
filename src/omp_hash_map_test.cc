#include "omp_hash_map.h"
#include "gtest/gtest.h"

TEST(OMPHashMap, Initialization) {
  cornell::omp_hash_map<std::string, double> m;
  EXPECT_EQ(m.get_n_keys(), 0);
}