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
        fs[c_i][d_i] = dis(prng);
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

    vector<vector<float>> bincount(nchains, vector<float>(depth));

    // compute bincounts
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
        bincount[c][d] = cmsketches[c][d][bin];
      }
    }

    // compute anomaly score
    float avg_anomalyscore = 0.0;
    for (uint c = 0; c < nchains; c++) {
      float score_c = bincount[c][0] * pow(2.0, 1);
      for (uint d = 1; d < depth; d++) {
        //if (bincount[c][d] < 25)
        //  break;
        float scaled_bincount = bincount[c][d] * pow(2.0, d+1);
        if (scaled_bincount < score_c)
          score_c = scaled_bincount;
      }
      avg_anomalyscore += score_c;
    }
    avg_anomalyscore /= nchains;

    return avg_anomalyscore;
  }

  tuple<vector<float>,vector<float>,float,float>
  chains_add2(vector<float>& x, vector<string>& feature_names,
             vector<uint64_t>& h, float density, float constant,
             vector<vector<float>>& deltamax, vector<vector<float>>& shift,
             vector<vector<unordered_map<vector<int>,int>>>& cmsketches,
             vector<vector<uint>>& fs, vector<vector<float>>& mean_bincount,
             float npoints, bool update) {

    uint k = h.size();
    uint nchains = cmsketches.size();
    uint depth = cmsketches[0].size();

    vector<vector<float>> bincount(nchains, vector<float>(depth));
    vector<vector<float>> lociscore(nchains, vector<float>(depth));
    vector<float> anomalyscore(nchains, numeric_limits<float>::max());
    vector<float> avg_bincount(depth);
    vector<float> avg_lociscore(depth);
    float avg_anomalyscore = 0.0;

    vector<float> xp = streamhash_project(x, feature_names, h, density,
                                          constant);

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

        int N = npoints;
        if (update) {
          mean_bincount[c][d] += 1.0 + 2.0 * cmsketches[c][d][bin];
          cmsketches[c][d][bin]++;
          N = npoints + 1;
        }

        bincount[c][d] = cmsketches[c][d][bin];
        lociscore[c][d] = (bincount[c][d] - (1.0/N) * mean_bincount[c][d])
                            * pow(2.0, d+1);

        avg_bincount[d] += bincount[c][d];
        avg_lociscore[d] += lociscore[c][d];
        if (bincount[c][d] < anomalyscore[c] * pow(2.0, d+1)) {
          anomalyscore[c] = bincount[c][d] * pow(2.0, d+1);
        }
      }
      avg_anomalyscore += anomalyscore[c];
    }

    if (update) { npoints += 1; }

    avg_anomalyscore /= nchains;
    for (uint d = 0; d < depth; d++) {
      avg_bincount[d] /= nchains;
      avg_lociscore[d] /= nchains;
    }

    avg_anomalyscore = 0.0;
    for (uint c = 0; c < nchains; c++) {
      float score_c = numeric_limits<float>::max();
      //float score_c = bincount[c][0] * pow(2.0, 1);
      for (uint d = 1; d < depth; d++) {
        if (bincount[c][d] * pow(2.0, d+1) < score_c)
          score_c = bincount[c][d] * pow(2.0, d+1);
        //if (bincount[c][d] < 25)
        //  break;
        //score_c = bincount[c][d] * pow(2.0, d+1);
      }
      avg_anomalyscore += score_c;
    }
    avg_anomalyscore /= nchains;

    return make_tuple(avg_bincount, avg_lociscore, avg_anomalyscore, npoints);
  }

}
