#include <cassert>
#include "chain.h"
#include <chrono>
#include "docopt.h"
#include "hash.h"
#include <iomanip>
#include <iostream>
#include <limits>
#include "param.h"
#include <random>
#include "streamhash.h"
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include "util.h"
#include <vector>

using namespace std;

static const char USAGE[] =
R"(xstream.

    Usage:
      xstream [--k=<projection size>]
              [--c=<number of chains>]
              [--d=<depth>]
              [--rowstream]
              [--nwindows=<windowed>]
              [--initsample=<initial sample size>]
              [--scoringbatch=<scoring batch size>]
              [--cosine]

      xstream (-h | --help)

    Options:
      -h, --help                           Show this screen.
      --k=<projection size>                Projection size [default: 100].
      --c=<number of chains>               Number of chains [default: 100].
      --d=<depth>                          Depth [default: 15].
      --rowstream                          Row stream (each row starts with a label).
      --nwindows=<number of windows>       > 0 if windowed [default: 1].
      --initsample=<initial sample size>   Initial sample size [default: 256].
      --scoringbatch=<scoring batch size>  Print scores at regular intervals [default: 1000].
      --cosine                             Work in cosine space instead of Euclidean.
)";

tuple<vector<vector<float>>,vector<vector<float>>>
compute_deltamax(vector<vector<float>>& window, uint c, uint k, mt19937_64& prng);

int main(int argc, char *argv[]) {
  // utility elements
#ifdef SEED
  mt19937_64 prng(SEED);
#else
  random_device rd;
  mt19937_64 prng(rd());
#endif

  // arguments
  map<string, docopt::value> args = docopt::docopt(USAGE, { argv + 1, argv + argc });

  // for timing
  chrono::time_point<chrono::steady_clock> start;
  chrono::time_point<chrono::steady_clock> end;
  chrono::milliseconds diff;

  // store arguments
  uint k = args["--k"].asLong();
  uint c = args["--c"].asLong();
  uint d = args["--d"].asLong();
  bool fixed = args["--rowstream"].asBool();
  bool cosine = args["--cosine"].asBool();

  int nwindows = args["--nwindows"].asLong(); // no windows by default
  uint init_sample_size = args["--initsample"].asLong(); // full data size by default
  uint scoring_batch_size = args["--scoringbatch"].asLong(); // 1000 by default

  cerr << "xstream: " << "K=" << k << " C=" << c << " d=" << d << " ";
  if (fixed) cerr << "(row-stream)";
  else cerr << "(evolving-stream)";

  if (nwindows > 0) cerr << " (windowed)";
  cerr << endl;
  cerr << "\tinitial sample: " << init_sample_size << " tuples" << endl;
  cerr << "\tscoring every: " << scoring_batch_size << " tuples" << endl;

  // initialize chains
  cerr << "initializing... ";
  start = chrono::steady_clock::now();

  // not used if working in cosine space
  vector<vector<float>> deltamax(c, vector<float>(k, 0.0));
  vector<vector<float>> shift(c, vector<float>(k, 0.0));
  
  vector<vector<unordered_map<vector<int>,int>>> cmsketches(c,
                                                          vector<unordered_map<vector<int>,int>>(d));
  vector<vector<uint>> fs(c, vector<uint>(d, 0));

  // initialize streamhash functions
  vector<uint64_t> h(k, 0);
  float density_constant = streamhash_compute_constant(DENSITY, k);

#ifdef VERBOSE
  string s("test");
  cout << "Empirical density = ";
  cout << streamhash_empirical_density(s, prng, DENSITY, density_constant) << endl;
#endif

  streamhash_init_seeds(h, prng);
  chains_init_features(fs, k, prng);

  end = chrono::steady_clock::now();
  diff = chrono::duration_cast<chrono::milliseconds>(end - start);
  cerr << "done in " << diff.count() << "ms" << endl;

  // current window of projected tuples, part of the model
  vector<vector<float>> window(init_sample_size, vector<float>(k));

  vector<float> anomalyscores;
  anomalyscores.reserve(1000000);

  // auxilliary data structures for timing
  vector<float> projection_times;
  vector<float> update_times;
  vector<float> tuple_times;

  // input tuples
  cerr << "streaming tuples from stdin..." << endl;
  start = chrono::steady_clock::now();
  stringstream ss;
  string line;
  uint row_idx = 1;
  uint window_size = 0;
  while (getline(cin, line)) {
    ss.clear();
    ss.str(line);

    string label_or_id;
    ss >> label_or_id;

    vector<string> fields;
    string field;
    while (ss >> field) { fields.push_back(field); }

    // project tuple
    vector<float> xp;
    //chrono::time_point<chrono::steady_clock> start = chrono::steady_clock::now();
    xp = streamhash_project(fields, h, DENSITY, density_constant);
    //chrono::time_point<chrono::steady_clock> end = chrono::steady_clock::now();
    //chrono::microseconds diff = chrono::duration_cast<chrono::microseconds>(end - start);
    //cout << "p " << setprecision(12) << static_cast<float>(diff.count())/fields.size() << endl;

    // if the initial sample has not been seen yet, continue
    if (row_idx < init_sample_size) {
      window[window_size] = xp;
      window_size++;
      row_idx++;
      continue;
    }

    // check if the initial sample just arrived
    if (row_idx == init_sample_size) {
      window[window_size] = xp;
      window_size++;

      if (!cosine) {
        // compute deltmax/shift from initial sample
        cerr << "initializing deltamax from sample size " << window_size << "..." << endl;
        tie(deltamax, shift) = compute_deltamax(window, c, k, prng);
      }

      // add initial sample tuples to chains
      for (auto x : window) {
        if (cosine) {
          chains_add_cosine(x, cmsketches, fs, true);
        } else {
          chains_add(x, deltamax, shift, cmsketches, fs, true);
        }
      }

      // score initial sample tuples
      cerr << "scoring first batch of " << init_sample_size << " tuples... ";
      for (auto x : window) {
        float anomalyscore;
        if (cosine) {
          anomalyscore = chains_add_cosine(x, cmsketches, fs, false);
        } else {
          anomalyscore = chains_add(x, deltamax, shift, cmsketches, fs, false);
        }
        anomalyscores.push_back(anomalyscore);
      }

      window_size = 0;
      row_idx++;
      cerr << "done." << endl;
      continue;
    }

    // row_idx > init_sample_size

    if (nwindows <= 0) { // non-windowed mode

      float anomalyscore;
      if (cosine) {
        anomalyscore = chains_add_cosine(xp, cmsketches, fs, true);
      } else {
        anomalyscore = chains_add(xp, deltamax, shift, cmsketches, fs, true);
      }
      anomalyscores.push_back(anomalyscore);

    } else if (nwindows > 0) { // windowed mode
      window[window_size] = xp;
      window_size++;

      float anomalyscore;
      if (cosine) {
        anomalyscore = chains_add_cosine(xp, cmsketches, fs, false);
      } else {
        anomalyscore = chains_add(xp, deltamax, shift, cmsketches, fs, false);
      }
      anomalyscores.push_back(anomalyscore);

      // if the batch limit is reached, construct new chains
      // while different from the paper, this is more cache-efficient
      if (window_size == static_cast<uint>(init_sample_size)) {
        cerr << "\tnew chains at tuple: " << row_idx << endl;

        // uncomment this to compute a new deltamax, shift from the new window points
        //tie(deltamax, shift) = compute_deltamax(window, c, k, prng);

        // clear old bincounts
        for (uint chain = 0; chain < c; chain++) {
          for (uint depth = 0; depth < d; depth++) {
            cmsketches[chain][depth].clear();
          }
        }

        // add current window tuples to chains
        for (auto x : window) {
          if (cosine) {
            chains_add_cosine(x, cmsketches, fs, true);
          } else {
            chains_add(x, deltamax, shift, cmsketches, fs, true);
          }
        }

        window_size = 0;
      }
    }

    if ((row_idx > init_sample_size) && (row_idx % scoring_batch_size == 0)) {
      cerr << "\tscoring at tuple: " << row_idx << endl;
      cout << row_idx << "\t";
      for (uint i = 0; i < anomalyscores.size(); i++) {
        float anomalyscore = anomalyscores[i];
        cout << setprecision(12) << anomalyscore << " ";
      }
      cout << endl;
    }

    row_idx++;
  }
  end = chrono::steady_clock::now();
  diff = chrono::duration_cast<chrono::milliseconds>(end - start);
  cerr << "done in " << diff.count() << "ms" << endl;

  // print tuple scores at the end 
  cerr << "final scores of " << anomalyscores.size() << " tuples... ";
  start = chrono::steady_clock::now();
  cout << anomalyscores.size() << "\t";
  for (uint i = 0; i < anomalyscores.size(); i++) {
    float anomalyscore = anomalyscores[i];
    cout << setprecision(12) << anomalyscore << " ";
  }
  cout << endl;
  end = chrono::steady_clock::now();
  diff = chrono::duration_cast<chrono::milliseconds>(end - start);
  cerr << "done in " << diff.count() << "ms" << endl;
}

tuple<vector<vector<float>>,vector<vector<float>>>
compute_deltamax(vector<vector<float>>& window, uint c, uint k, mt19937_64& prng) {

  vector<vector<float>> deltamax(c, vector<float>(k, 0.0));
  vector<vector<float>> shift(c, vector<float>(k, 0.0));

  vector<float> dim_min(k, numeric_limits<float>::max());
  vector<float> dim_max(k, numeric_limits<float>::min());

  for (auto x : window) {
    for (uint j = 0; j < k; j++) {
      if (x[j] > dim_max[j]) { dim_max[j] = x[j]; }
      if (x[j] < dim_min[j]) { dim_min[j] = x[j]; }
    }
  }

  // initialize deltamax to half the projection range, shift ~ U(0, dmax)
  for (uint i = 0; i < c; i++) {
    for (uint j = 0; j < k; j++) {
      deltamax[i][j] = (dim_max[j] - dim_min[j])/2.0;
      if (abs(deltamax[i][j]) <= EPSILON) {
        deltamax[i][j] = 1.0;
      }
      uniform_real_distribution<> dis(0, deltamax[i][j]);
      shift[i][j] = dis(prng);
    }
  }

  return make_tuple(deltamax, shift);
}
