/*
 * See "Practical Hash Functions for Similarity Estimation
 * and Dimensionality Reduction" for a study on various hash
 * functions: MurmurHash3 works as well as a theoretically
 * "regular" hash function proposed in (Dahlgaard, FOCS 2015).
 */
#include <cstring>
#include <iostream>
#include <limits>
#include <math.h>
#include "MurmurHash3.h"
#include "param.h"
#include <random>
#include <vector>

namespace std {

  void
  streamhash_init_seeds(vector<uint64_t>& h, mt19937_64& prng) {
    uint k = h.size();
    for (uint k_i = 0; k_i < k; k_i++) {
      h[k_i] = prng();
    }
  }

  float
  streamhash_compute_constant(float density, uint k) {
    return sqrt(static_cast<double>(1.0/(density * k)));
  }

  float
  streamhash_hash(string& s, uint64_t seed, float density, float constant) {
    uint64_t hash_value[2]; // 128 bits
    int len = s.length();

    MurmurHash3_x64_128(&s, len, seed, &hash_value);

    float hash_value_f = static_cast<float>(hash_value[0]);
    float max_value_f = static_cast<float>((uint64_t)-1);
    hash_value_f = hash_value_f/max_value_f;

    if (hash_value_f <= density/2.0) {
      return -1.0 * constant;
    } else if (hash_value_f <= density) {
      return constant;
    } else {
      return 0.0;
    }
  }

  float
  streamhash_hash(uint32_t s, uint64_t seed, float density, float constant) {
    uint64_t hash_value[2]; // 128 bits
    int len = 4; // 4 bytes = 32 bits

    MurmurHash3_x64_128(&s, len, seed, &hash_value);

    float hash_value_f = static_cast<float>(hash_value[0]);
    float max_value_f = static_cast<float>((uint64_t)-1);
    hash_value_f = hash_value_f/max_value_f;

    if (hash_value_f <= density/2.0) {
      return -1.0 * constant;
    } else if (hash_value_f <= density) {
      return constant;
    } else {
      return 0.0;
    }
  }

  float
  streamhash_empirical_density(string& s, mt19937_64& prng,
                               float density, float constant) {
    int ntrials = 100000;
    float nonzeros = 0.0;
    for (int i = 0; i < ntrials; i++) {
      if (abs(streamhash_hash(s, prng(), density, constant)) > EPSILON) {
        nonzeros++;
      }
    }
    return nonzeros/ntrials; 
  }

  vector<float>
  streamhash_project(vector<string>& fields,
                     vector<uint64_t>& h, float density, float constant) {
    uint k = h.size();
    vector<float> xp(k, 0.0);
    for (uint i = 0; i < k; i++) {
      for (uint j = 0; j < fields.size(); j++) {
        // format: f1:v1 f2:v2 ... fk:vk, k <= D
        uint32_t feature_idx;
        float feature_value;
        const char *fields_str = fields[j].c_str();
        char *dup = strdup(fields_str);
        feature_idx = static_cast<uint32_t>(atoi(strtok(dup, ":")));
        feature_value = atof(strtok(NULL, ":"));
        free(dup);
        xp[i] += (feature_value * streamhash_hash(feature_idx, h[i], density, constant));
      }
    }
    return xp;
  }

  vector<float>
  streamhash_project(vector<float>& x, vector<string>& feature_names,
                     vector<uint64_t>& h, float density, float constant) {
    uint k = h.size();
    uint dims = x.size();
    vector<float> xp(k, 0.0);
    for (uint i = 0; i < k; i++) {
      for (uint j = 0; j < dims; j++) {
        xp[i] += (x[j] *  streamhash_hash(feature_names[j], h[i],
                                          density, constant));
      }
    }
    return xp;
  }

}
