#include "omp_hash_map.h"
#include "reducer.h"
#include "gtest/gtest.h"
#include "omp.h"

TEST(OMPHashMapTest, Initialization) {
  cornell::hpc::omp_hash_map<std::string, int> m;
  EXPECT_EQ(m.get_n_keys(), 0);
}

TEST(OMPHashMapTest, Reserve) {
  // Explicit reserve tests.
  cornell::hpc::omp_hash_map<std::string, int> m;
  m.reserve(10);
  EXPECT_GE(m.get_n_buckets(), 10);

  // Automatic reserve tests.
  cornell::hpc::omp_hash_map<int, int> m2;
  for (int i = 0; i < 100; i++) {
    m2.set(i, i * i);
    EXPECT_EQ(m2.get_n_keys(), i + 1);
    EXPECT_GE(m2.get_n_buckets(), i + 1);
  }
  for (int i = 0; i < 100; i++) {
    EXPECT_EQ(m2.get_copy_or_default(i, 0), i * i);
  }
}

TEST(OMPHashMapLargeTest, FourBillionsReserve) {
  cornell::hpc::omp_hash_map<std::string, int> m;
  constexpr size_t LARGE_N_BUCKETS = 4000000000;
  m.reserve(LARGE_N_BUCKETS);
  const size_t n_buckets = m.get_n_buckets();
  EXPECT_GE(n_buckets, LARGE_N_BUCKETS);
}

TEST(OMPHashMapTest, Set) {
  cornell::hpc::omp_hash_map<std::string, int> m;
  // Set with value.
  m.set("aa", 1);
  EXPECT_EQ(m.get_copy_or_default("aa", 0), 1);

  // Set with setter function.
  const auto& increase_by_one = [&](auto& value) { value++; };
  m.set("aa", increase_by_one);
  EXPECT_EQ(m.get_copy_or_default("aa", 0), 2);

  // Set with setter function and a custom default value.
  m.set("aa", increase_by_one, 0);
  EXPECT_EQ(m.get_copy_or_default("aa", 0), 3);
  m.set("bbb", increase_by_one, 5);
  EXPECT_EQ(m.get_copy_or_default("bbb", 0), 6);
}

TEST(OMPHashMapTest, Unset) {
  cornell::hpc::omp_hash_map<std::string, int> m;
  m.set("aa", 1);
  m.set("bbb", 2);
  m.unset("aa");
  EXPECT_FALSE(m.has("aa"));
  EXPECT_TRUE(m.has("bbb"));
  EXPECT_EQ(m.get_n_keys(), 1);

  m.unset("not_exist_key");
  EXPECT_EQ(m.get_n_keys(), 1);

  m.unset("bbb");
  EXPECT_FALSE(m.has("aa"));
  EXPECT_FALSE(m.has("bbb"));
  EXPECT_EQ(m.get_n_keys(), 0);
}

TEST(OMPHashMapTest, Map) {
  cornell::hpc::omp_hash_map<std::string, int> m;
  const auto& cubic = [&](const int value) { return value * value * value; };
  m.set("aa", 5);
  EXPECT_EQ(m.map<int>("aa", cubic, 0), 125);
  EXPECT_EQ(m.map<int>("not_exist_key", cubic, 3), 3);
}

TEST(OMPHashMapTest, Apply) {
  cornell::hpc::omp_hash_map<std::string, int> m;
  m.set("aa", 5);
  m.set("bbb", 10);
  int sum = 0;

  // Apply to one key.
  m.apply("aa", [&](const auto& value) { return sum += value; });
  EXPECT_EQ(sum, 5);

  // Apply to all the keys.
  m.apply([&](const auto& key, const auto& value) {
    if (key.front() == 'b') sum += value;
  });
  EXPECT_EQ(sum, 15);
}

TEST(OMPHashMapTest, MapReduce) {
  cornell::hpc::omp_hash_map<std::string, double> m;
  m.set("aa", 1.1);
  m.set("ab", 2.2);
  m.set("ac", 3.3);
  m.set("ad", 4.4);
  m.set("ae", 5.5);
  m.set("ba", 6.6);
  m.set("bb", 7.7);
  // Count the number of keys that start with 'a'.
  const auto& initial_a_to_one = [&](const std::string& key, const auto value) {
    (void)value;  // Prevent unused variable warning.
    if (key.front() == 'a') return 1;
    return 0;
  };
  const int initial_a_count = m.map_reduce<int>(initial_a_to_one, cornell::hpc::reducer::sum<int>, 0);
  EXPECT_EQ(initial_a_count, 5);
}

TEST(OMPHashMapLargeTest, QuarterBillionsMapReduce) {
  cornell::hpc::omp_hash_map<int, int> m;
  constexpr int LARGE_N_KEYS = 250000000;

  m.reserve(LARGE_N_KEYS);
#pragma omp parallel for schedule(static, 1)
  for (int i = 0; i < LARGE_N_KEYS; i++) {
    m.set(i, i);
  }
  const auto& mapper = [&](const int key, const int value) {
    (void)key;
    return value;
  };
  const auto& sum = m.map_reduce<int>(mapper, cornell::hpc::reducer::max<int>, 0.0);
  EXPECT_EQ(sum, LARGE_N_KEYS - 1);
}

TEST(OMPHashMapTest, Clear) {
  cornell::hpc::omp_hash_map<std::string, int> m;
  m.set("aa", 1);
  m.set("bbb", 2);
  m.clear();
  EXPECT_FALSE(m.has("aa"));
  EXPECT_FALSE(m.has("bbb"));
  EXPECT_EQ(m.get_n_keys(), 0);
}