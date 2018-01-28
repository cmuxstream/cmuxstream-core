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
  chains_add(vector<float>& x, vector<string>& feature_names,
             vector<uint64_t>& h, float density, float constant,
             vector<vector<float>>& deltamax, vector<vector<float>>& shift,
             vector<vector<unordered_map<vector<int>,int>>>& cmsketches,
             vector<vector<uint>>& fs, bool update);

  tuple<vector<float>,vector<float>,float,float>
  chains_add2(vector<float>& x, vector<string>& feature_names,
             vector<uint64_t>& h, float density, float constant,
             vector<vector<float>>& deltamax, vector<vector<float>>& shift,
             vector<vector<unordered_map<vector<int>,int>>>& cmsketches,
             vector<vector<uint>>& fs, vector<vector<float>>& mean_bincount,
             float npoints, bool update);
}

#endif
