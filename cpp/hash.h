#ifndef XSTREAM_HASH_H_
#define XSTREAM_HASH_H_

#include <string>
#include <vector>

namespace std {

/* Combination hash from Boost */
template <class T>
inline void hash_combine(size_t& seed, const T& v)
{
    hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template<typename T> struct hash<vector<T>>
{
  inline size_t operator()(const vector<T>& v) const
  {
    size_t seed = 0;
    for (uint i = 0; i < v.size(); i++)
      hash_combine(seed, v[i] * 2654435761);
    return seed;
  }
};
/* End combination hash from Boost */

}

#endif
