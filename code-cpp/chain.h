#ifndef XSTREAM_CHAIN_H_
#define XSTREAM_CHAIN_H_

#include <random>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace std {

  void
  chains_init_features(vector<vector<uint>>& fs, uint k, mt19937_64& prng);
  
  void
  chains_init_deltamax(vector<vector<float>>& X,
                       uint sample_size,
                       vector<vector<float>>& deltamax,
                       vector<vector<float>>& shift,
                       mt19937_64& prng);

}

#endif
