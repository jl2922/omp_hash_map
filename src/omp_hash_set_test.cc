#include "omp_hash_set.h"
#include "gtest/gtest.h"
#include "omp.h"
#include "reducer.h"

TEST(OMPHashSetTest, Initialization) {
  omp_hash_set<std::string> m;
  EXPECT_EQ(m.get_n_keys(), 0);
}

TEST(OMPHashSetTest, Reserve) {
  // Explicit reserve tests.
  omp_hash_set<std::string> m;
  m.reserve(10);
  EXPECT_GE(m.get_n_buckets(), 10);

  // Automatic reserve tests.
  omp_hash_set<int> m2;
  for (int i = 0; i < 100; i++) {
    m2.add(i);
    EXPECT_EQ(m2.get_n_keys(), i + 1);
    EXPECT_GE(m2.get_n_buckets(), i + 1);
  }
  for (int i = 0; i < 100; i++) {
    EXPECT_TRUE(m2.has(i));
  }
}

TEST(OMPHashSetTest, OneMillionReserve) {
  omp_hash_set<std::string> m;
  constexpr size_t LARGE_N_BUCKETS = 1000000;
  m.reserve(LARGE_N_BUCKETS);
  const size_t n_buckets = m.get_n_buckets();
  EXPECT_GE(n_buckets, LARGE_N_BUCKETS);
}

TEST(OMPHashSetLargeTest, HundredMillionsReserve) {
  omp_hash_set<std::string> m;
  constexpr size_t LARGE_N_BUCKETS = 100000000;
  m.reserve(LARGE_N_BUCKETS);
  const size_t n_buckets = m.get_n_buckets();
  EXPECT_GE(n_buckets, LARGE_N_BUCKETS);
}

TEST(OMPHashSetTest, Add) {
  omp_hash_set<std::string> m;
  // Set with value.
  m.add("aa");
  EXPECT_TRUE(m.has("aa"));
  m.add("aa");
  EXPECT_TRUE(m.has("aa"));

  m.add("bbb");
  EXPECT_TRUE(m.has("aa"));
  EXPECT_TRUE(m.has("bbb"));
  EXPECT_FALSE(m.has("not_exist_key"));
}

TEST(OMPHashSetLargeTest, TenMillionsInsertWithAutoRehash) {
  omp_hash_set<int> m;
  constexpr int LARGE_N_KEYS = 10000000;

  omp_set_nested(1);  // Parallel rehashing.
#pragma omp parallel for
  for (int i = 0; i < LARGE_N_KEYS; i++) {
    m.add(i);
  }
  EXPECT_EQ(m.get_n_keys(), LARGE_N_KEYS);
  EXPECT_GE(m.get_n_buckets(), LARGE_N_KEYS);
}

TEST(OMPHashSetTest, Remove) {
  omp_hash_set<std::string> m;
  m.add("aa");
  m.add("bbb");
  m.remove("aa");
  EXPECT_FALSE(m.has("aa"));
  EXPECT_TRUE(m.has("bbb"));
  EXPECT_EQ(m.get_n_keys(), 1);

  m.remove("not_exist_key");
  EXPECT_EQ(m.get_n_keys(), 1);

  m.remove("bbb");
  EXPECT_FALSE(m.has("aa"));
  EXPECT_FALSE(m.has("bbb"));
  EXPECT_EQ(m.get_n_keys(), 0);
}

TEST(OMPHashSetTest, Apply) {
  omp_hash_set<std::string> m;
  m.add("aa");
  m.add("bbb");
  int initial_a_count = 0;

  // Apply to all the keys.
  m.apply([&](const auto& key) {
    if (key.front() == 'a') initial_a_count++;
  });
  EXPECT_EQ(initial_a_count, 1);
}

TEST(OMPHashSetTest, MapReduce) {
  omp_hash_set<std::string> m;
  m.add("aa");
  m.add("ab");
  m.add("ac");
  m.add("ad");
  m.add("ae");
  m.add("ba");
  m.add("bb");
  // Count the number of keys that start with 'a'.
  const auto& initial_a_to_one = [&](const std::string& key) {
    if (key.front() == 'a') return 1;
    return 0;
  };
  const int initial_a_count = m.map_reduce<int>(initial_a_to_one, reducer::sum<int>, 0);
  EXPECT_EQ(initial_a_count, 5);
}

TEST(OMPHashSetLargeTest, TenMillionsMapReduce) {
  omp_hash_set<int> m;
  constexpr int LARGE_N_KEYS = 10000000;

  m.reserve(LARGE_N_KEYS);
#pragma omp parallel for
  for (int i = 0; i < LARGE_N_KEYS; i++) {
    m.add(i);
  }
  const auto& mapper = [&](const int key) { return key; };
  const auto& sum = m.map_reduce<int>(mapper, reducer::max<int>, 0.0);
  EXPECT_EQ(sum, LARGE_N_KEYS - 1);
}

TEST(OMPHashSetTest, Clear) {
  omp_hash_set<std::string> m;
  m.add("aa");
  m.add("bbb");
  m.clear();
  EXPECT_FALSE(m.has("aa"));
  EXPECT_FALSE(m.has("bbb"));
  EXPECT_EQ(m.get_n_keys(), 0);
}