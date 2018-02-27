#include <bitset>
#include <cassert>
#include "chain.h"
#include <cmath>
#include "hash.h"
#include <iostream>
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
    uniform_int_distribution<> dis(0, k-1);

    for (uint c_i = 0; c_i < c; c_i++) {
      for (uint d_i = 0; d_i < d; d_i++) {
        int feature = dis(prng);
        fs[c_i][d_i] = feature;
      }
    }
  }

  float
  chains_add(vector<float>& xp, vector<vector<float>>& deltamax, vector<vector<float>>& shift,
             vector<vector<unordered_map<vector<int>,int>>>& cmsketches,
             vector<vector<uint>>& fs, bool update) {

    uint k = xp.size();
    uint nchains = cmsketches.size();
    uint depth = cmsketches[0].size();

    vector<vector<float>> scaled_bincount(nchains, vector<float>(depth));

    for (uint c = 0; c < nchains; c++) {
      vector<float> prebin(k, 0.0);
      vector<bool> used(k, false);
      for (uint d = 0; d < depth; d++) {
        uint f = fs[c][d];
        if (used[f] == false) {
          prebin[f] = (xp[f] + shift[c][f])/deltamax[c][f];
          used[f] = true;
        } else {
          prebin[f] = 2.0 * prebin[f] - shift[c][f]/deltamax[c][f];
        }

        vector<int> bin(k);
        for (uint i = 0; i < k; i++) {
          bin[i] = static_cast<int>(floor(prebin[i]));
        }

        if (update) {
          cmsketches[c][d][bin]++;
        }
        scaled_bincount[c][d] = log2(cmsketches[c][d][bin] + 1) + (d + 1);
      }
    }

    float avg_anomalyscore = 0.0;
    for (uint c = 0; c < nchains; c++) {
      float score_c = scaled_bincount[c][0];
      for (uint d = 1; d < depth; d++) {
        if (scaled_bincount[c][d] < score_c) {
          score_c = scaled_bincount[c][d];
        }
      }
      avg_anomalyscore += score_c;
    }
    avg_anomalyscore /= nchains;

    return avg_anomalyscore;
  }

  float
  chains_add_cosine(vector<float>& xp,
                    vector<vector<unordered_map<vector<int>,int>>>& cmsketches,
                    vector<vector<uint>>& fs, bool update) {

    uint nchains = cmsketches.size();
    uint depth = cmsketches[0].size();

    vector<vector<float>> scaled_bincount(nchains, vector<float>(depth));

    for (uint c = 0; c < nchains; c++) {
      vector<int> bin;
      for (uint d = 0; d < depth; d++) {
        bin.push_back(signbit(xp[fs[c][d]]));
        if (update) {
          cmsketches[c][d][bin]++;
        }
        scaled_bincount[c][d] = log2(cmsketches[c][d][bin] + 1) + (d + 1);
      }
    }

    float avg_anomalyscore = 0.0;
    for (uint c = 0; c < nchains; c++) {
      float score_c = scaled_bincount[c][0];
      for (uint d = 1; d < depth; d++) {
        if (scaled_bincount[c][d] < score_c) {
          score_c = scaled_bincount[c][d];
        }
      }
      avg_anomalyscore += score_c;
    }
    avg_anomalyscore /= nchains;

    return avg_anomalyscore;
  }
}
