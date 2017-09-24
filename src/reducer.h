#include <functional>

namespace reducer {
template <class T>
const std::function<void(T&, const T&)> sum = [](T& t1, const T t2) { t1 += t2; };

template <class T>
const std::function<void(T&, const T&)> max = [](T& t1, const T t2) {
  if (t1 < t2) t1 = t2;
};

template <class T>
const std::function<void(T&, const T&)> min = [](T& t1, const T t2) {
  if (t1 > t2) t1 = t2;
};
}
