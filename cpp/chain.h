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

  float
  chains_add(vector<float>& xp, vector<vector<float>>& deltamax, vector<vector<float>>& shift,
             vector<vector<unordered_map<vector<int>,int>>>& cmsketches,
             vector<vector<uint>>& fs, bool update);

  float
  chains_add_cosine(vector<float>& xp,
                    vector<vector<unordered_map<vector<int>,int>>>& cmsketches,
                    vector<vector<uint>>& fs, bool update);
}

#endif
