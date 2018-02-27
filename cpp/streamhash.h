#ifndef XSTREAM_STREAMHASH_H_
#define XSTREAM_STREAMHASH_H_

#include <random>
#include "MurmurHash3.h"
#include <vector>

namespace std {

  void
  streamhash_init_seeds(vector<uint64_t>& h, mt19937_64& prng);

  float
  streamhash_compute_constant(float density, uint k);

  float
  streamhash_hash(string& s, uint64_t seed, float density, float constant);

  float
  streamhash_hash(uint32_t& i, uint64_t seed, float density, float constant);

  float
  streamhash_empirical_density(string& s, mt19937_64& prng, float density,
                               float constant);

  vector<float>
  streamhash_project(vector<float>& x, vector<string>& feature_names,
                     vector<uint64_t>& h, float density, float constant);

  vector<float>
  streamhash_project(vector<string>& fields,
                     vector<uint64_t>& h, float density, float constant);
}
#endif
