#ifndef XSTREAM_UTIL_H_
#define XSTREAM_UTIL_H_

#include <cmath>
#include <string>
#include <vector>
#include <iostream>

namespace std {

  inline void
  panic(string message) {
    cout << message << endl;
    exit(-1);
  }

  float
  euclidean_distance(vector<float>& x, vector<float>& y) {
    float dist = 0.0;
    uint n = x.size();
    for (uint i = 0; i < n; i++)
      dist += (x[i] - y[i]) * (x[i] - y[i]);
    dist = sqrt(dist);
    return dist;
  }

}

#endif
