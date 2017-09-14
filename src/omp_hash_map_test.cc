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

TEST(OMPHashMap, UnsetAndCheck) {
  cornell::omp_hash_map<std::string, double> m;
  m.set("aa", 1);
  m.set("bbb", 2);
  EXPECT_TRUE(m.has("aa"));
  EXPECT_TRUE(m.has("bbb"));
  m.unset("aa");
  EXPECT_FALSE(m.has("aa"));
  EXPECT_TRUE(m.has("bbb"));
  m.unset("not_exist_key");
}

TEST(OMPHashMap, Clear) {
  cornell::omp_hash_map<std::string, double> m;
  m.set("aa", 1);
  m.set("bbb", 2);
  EXPECT_TRUE(m.has("aa"));
  EXPECT_TRUE(m.has("bbb"));
  m.clear();
  EXPECT_FALSE(m.has("aa"));
  EXPECT_FALSE(m.has("bbb"));
}

TEST(OMPHashMap, MapReduce) {
  cornell::omp_hash_map<std::string, double> m;
  m.set("aa", 1);
  m.set("ab", 2);
  m.set("ac", 3);
  m.set("ad", 4);
  m.set("ae", 5);
  m.set("ba", 6);
  m.set("bb", 7);
  // Count the number of keys that start with 'a'.
  const auto& initial_a_to_one = [&](const std::string& key, const double value) {
    (void)value;  // Prevent unused variable warning.
    if (key.front() == 'a') return 1;
    return 0;
  };
  const auto& plus = [&](size_t& old_value, const size_t new_value) { old_value += new_value; };
  const size_t initial_a_count = m.map_reduce<size_t>(initial_a_to_one, plus, 0);
  EXPECT_EQ(initial_a_count, 5);
}