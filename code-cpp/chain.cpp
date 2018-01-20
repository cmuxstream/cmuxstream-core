#include <cassert>
#include "chain.h"
#include <limits>
#include <random>
#include "streamhash.h"
#include <string>
#include <vector>

namespace std {

  void
  chains_init_features(vector<vector<uint>>& fs, uint k, mt19937_64& prng) {
    uint c = fs.size();
    uint d = fs[0].size();
    uniform_int_distribution<> dis(0, k);

    for (uint c_i = 0; c_i < c; c_i++) {
      for (uint d_i = 0; d_i < d; d_i++) {
        fs[c_i][d_i] = dis(prng);
      }
    }
  }

  tuple<int,float,float>
  chains_add(vector<float>& x, vector<string>& feature_names,
             vector<uint64_t>& h, float density, float constant,
             vector<vector<float>>& deltamax, vector<vector<float>>& shift,
             vector<vector<unordered_map<string,int>>>& cmsketches) {
    int bincount;
    float lociscore;
    float anomalyscore;

    return make_tuple(bincount, lociscore, anomalyscore);
  }

}
