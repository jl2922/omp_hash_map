#include "omp_hash_map.h"
#include "reducer.h"
#include "gtest/gtest.h"
#include "omp.h"

TEST(ReducerTest, ReduceSum) {
  cornell::hpc::omp_hash_map<int, int> m;
  for (int i = 0; i < 100; i++) m.set(i, i);
  const auto& get_value = [](const int key, const int value) {
    (void)key;
    return value;
  };
  EXPECT_EQ(m.map_reduce<int>(get_value, cornell::hpc::reducer::sum<int>, 0), 4950);
}

TEST(ReducerTest, ReduceMax) {
  cornell::hpc::omp_hash_map<int, int> m;
  for (int i = 0; i < 100; i++) m.set(i, i);
  const auto& get_value = [](const int key, const int value) {
    (void)key;
    return value;
  };
  EXPECT_EQ(m.map_reduce<int>(get_value, cornell::hpc::reducer::max<int>, 0), 99);
}

TEST(ReducerTest, ReduceMin) {
  cornell::hpc::omp_hash_map<int, int> m;
  for (int i = 0; i < 100; i++) m.set(i, i);
  const auto& get_value = [](const int key, const int value) {
    (void)key;
    return value;
  };
  EXPECT_EQ(m.map_reduce<int>(get_value, cornell::hpc::reducer::min<int>, 0), 0);
}