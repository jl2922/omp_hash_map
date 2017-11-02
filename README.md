# omp hash map
A high performance and thread-safe implementation of C++ hash maps and hash sets.

[![Build Status](https://travis-ci.org/jl2922/omp_hash_map.svg?branch=master&style=flat)](https://travis-ci.org/jl2922/omp_hash_map)
[![Coverage Status](https://coveralls.io/repos/github/jl2922/omp_hash_map/badge.svg?branch=master&style=flat)](https://coveralls.io/github/jl2922/omp_hash_map?branch=master)

## Features
- Thread-safe access and modification.
- Parallel rehashing and clearing.
- Parallel Map-reduce.
- Get and set in one shot.

## Usage

Omp hash map is pure template library defined in headers.
To use this library, include the corresponding header files, compile with the OpenMP support of the compiler enabled, and set the c++ standard to c++14 or newer.

## Example
```c++
#include "omp_hash_map.h"
#include "reducer.h"
#include <string>

int main() {
  omp_hash_map<std::string, int> m;

  // Set and get.
  m.set("a", 0);
  m.set("b", 1);
  m.set("c", 2);
  m.get_copy_or_default("b", 0);  // 1.
  const auto& complicated_function = [&](auto& value) { value++; };
  m.set("b", complicated_function);  // Only perform one hash node look up.
  m.get_copy_or_default("b", 0);  // 2.

  // Parallel Map reduce.
  const auto& square = [&](const int key, const int value) {
    return value * value;
  };
  const int sum = m.map_reduce<int>(square, reducer::sum<int>, 0);  // 8.

  return 0;

  // Automatic parallel clearing.
}
```

For more examples, check the test files in the source folder.