#include <cassert>
#include "chain.h"
#include <limits>
#include <random>
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

  vector<float>
  chain_project(vector<float>& x, vector<string> feature_names,
                vector<uint64_t>& h) {
    // TODO
    vector<float> xp;
    return xp;
  }

  vector<vector<float>>
  chain_project(vector<vector<float>>& X, vector<string> feature_names,
                vector<uint64_t>& h) {
    // TODO
    vector<vector<float>> Xp;
    return Xp;
  }

  void
  chains_init_deltamax(vector<vector<float>>& X,
                       uint sample_size,
                       vector<vector<float>>& deltamax,
                       vector<vector<float>>& shift,
                       mt19937_64& prng) {
    uint c = deltamax.size();
    uint nrows = X.size();
    uint ndims = X[0].size();
    vector<float> range(ndims, 0.0);

    for (uint dim = 0; dim < ndims; dim++) {
      float dim_max = numeric_limits<float>::min();
      float dim_min = numeric_limits<float>::max();
      for (uint row = 0; row < nrows; row++) {
        float x = X[row][dim];
        if (x > dim_max) { dim_max = x; }
        if (x < dim_min) { dim_min = x; }
      }
      range[dim] = dim_max - dim_min;
    }

    for (uint c_i = 0; c_i < c; c_i++) {
      for (uint dim = 0; dim < ndims; dim++) {
        deltamax[c_i][dim] = range[dim]/2.0;
        uniform_real_distribution<> dis(0, deltamax[c_i][dim]);
        shift[c_i][dim] = dis(prng);
      }
    }
  }

}
