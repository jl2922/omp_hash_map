#include "omp_hash_map.h"
#include "gtest/gtest.h"

TEST(OMPHashMap, Initialization) {
  cornell::omp_hash_map<std::string, double> m;
  EXPECT_EQ(m.get_n_keys(), 0);
}

TEST(OMPHashMap, SingleSetAndApply) {
  cornell::omp_hash_map<std::string, double> m;
  m.set("aa", 1);
  m.set("bbb", 2);
  double sum = 0;
  const auto& add_to_sum = [&](const double value) { sum += value; };
  m.apply("aa", add_to_sum);
  EXPECT_EQ(sum, 1);
  m.apply("bbb", add_to_sum);
  EXPECT_EQ(sum, 3);
  m.apply("not_exist_key", add_to_sum);
  EXPECT_EQ(sum, 3);
}