#include <cassert>
#include "chain.h"
#include <chrono>
#include "docopt.h"
#include "hash.h"
#include "io.h"
#include <iomanip>
#include <iostream>
#include <limits>
#include "param.h"
#include <random>
#include "streamhash.h"
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
      xstream --input=<input file>
              --k=<projection size>
              --c=<number of chains>
              --d=<depth> 
              [--fixed]
              [--nwindows=<number of windows>]
              [--initsample=<initial sample size>]
              [--scoringbatch=<scoring batch size>]
              [--score-once]

      xstream (-h | --help)

    Options:
      -h, --help                           Show this screen.
      --input=<input file>                 Input file.
      --k=<projection size>                Projection size.
      --c=<number of chains>               Number of chains.
      --d=<depth>                          Depth.
      --fixed                              Fixed feature space.
      --nwindows=<number of windows>       Number of windows [default: 0].
      --initsample=<initial sample size>   Initial sample size [default: -1].
      --scoringbatch=<scoring batch size>  Scoring batch size [default: 1000].
      --score-once                         Score each point only once.
)";

void print_scores(uint row_idx, vector<vector<float>>& X, vector<string>& feature_names,
                  vector<uint64_t>& h, float density, float density_constant,
                  vector<vector<float>>& deltamax, vector<vector<float>>& shift,
                  vector<vector<unordered_map<vector<int>,int>>> cmsketches, // copy
                  vector<vector<uint>>& fs, vector<vector<float>> mean_bincount, // copy
                  vector<float>& anomalyscores, bool score_once,
                  float npoints);

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

  // store argumetns
  string input_file(args["--input"].asString());
  uint k = args["--k"].asLong();
  uint c = args["--c"].asLong();
  uint d = args["--d"].asLong();
  bool fixed = args["--fixed"].asBool();
  bool score_once = args["--score-once"].asBool();

  int nwindows = args["--nwindows"].asLong(); // no windows by default
  int init_sample_size = args["--initsample"].asLong(); // full data size by default
  uint scoring_batch_size = args["--scoringbatch"].asLong(); // 1000 by default

  cerr << "xstream: "
       << "K=" << k << " C=" << c << " d=" << d
       << " windows=" << nwindows << endl;
  cerr << "\tinput: " << input_file << " ";
  if (fixed) cerr << "(fixed feature space)";
  cerr << endl;
  cerr << "\tscoring every: " << scoring_batch_size << " tuples" << endl;

  // initialize chains
  cerr << "initializing... ";
  start = chrono::steady_clock::now();

  vector<vector<float>> deltamax(c, vector<float>(k, 0.0));
  vector<vector<float>> shift(c, vector<float>(k, 0.0));
  vector<vector<unordered_map<vector<int>,int>>> cmsketches(c,
                                                          vector<unordered_map<vector<int>,int>>(d));
  vector<vector<uint>> fs(c, vector<uint>(d, 0));
  vector<uint64_t> h(k, 0);
  float density_constant = streamhash_compute_constant(DENSITY, k);
  vector<vector<float>> mean_bincount(c, vector<float>(d));
  float npoints = 0.0; // number of points

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

  // input stream
  if (fixed) {
    // fixed feature space

    // read input
    cerr << "Reading input tuples..." << endl;
    vector<vector<float>> X;
    vector<bool> Y;
    tie(X, Y) = input_fixed(input_file);
    uint nrows = X.size();
    uint ndims = X[0].size();

#ifdef VERBOSE
    for (uint i = 0; i < nrows; i++) {
      vector<float> x = X[i];
      for (const auto& v: x) {
        cout << v << ",";
      }
      cout << "\t" << Y[i] << endl;
    }
#endif 

    // construct auxiliary data structures
    vector<vector<float>> bincounts(nrows, vector<float>(d));
    vector<vector<float>> lociscores(nrows, vector<float>(d));
    vector<float> anomalyscores(nrows);

    // construct feature names
    vector<string> feature_names(ndims);
    for (uint i = 0; i < ndims; i++) {
      feature_names[i] = to_string(i);
    }

    // construct projection of an initial sample, compute projection range
    if (init_sample_size < 0) {
      init_sample_size = nrows;
    }
    cerr << "Initializing deltamax from sample size " << init_sample_size << "..." << endl;
    vector<vector<float>> Xpsample(init_sample_size, vector<float>(k, 0.0));
    vector<float> dim_min(k, numeric_limits<float>::max());
    vector<float> dim_max(k, numeric_limits<float>::min());
    for (int i = 0; i < init_sample_size; i++) {
      Xpsample[i] = streamhash_project(X[i], feature_names, h, DENSITY,
                                       density_constant);
      for (uint j = 0; j < k; j++) {
        if (Xpsample[i][j] > dim_max[j]) { dim_max[j] = Xpsample[i][j]; }
        if (Xpsample[i][j] < dim_min[j]) { dim_min[j] = Xpsample[i][j]; }
      }
    }

    // initialize deltamax to half the projection range, shift ~ U(0, dmax)
    for (uint i = 0; i < c; i++) {
      for (uint j = 0; j < k; j++) {
        deltamax[i][j] = (dim_max[j] - dim_min[j])/2.0;
        if (abs(deltamax[i][j]) <= EPSILON) deltamax[i][j] = 1.0;
        uniform_real_distribution<> dis(0, deltamax[i][j]);
        shift[i][j] = dis(prng);
      }
    }

    // stream in tuples
    vector<thread> scoring_threads;
    cerr << "streaming in " << nrows << " tuples... " << endl;
    start = chrono::steady_clock::now();
    for (uint row_idx = 0; row_idx < nrows; row_idx++) {
      //vector<float> bincount, lociscore;
      float anomalyscore;
      //tie(bincount, lociscore, anomalyscore, npoints) =
      tie(anomalyscore, npoints) =
        chains_add(X[row_idx], feature_names, h, DENSITY, density_constant, deltamax, shift,
                   cmsketches, fs, mean_bincount, npoints, true);

      anomalyscores[row_idx] = anomalyscore;
      if ((row_idx > 0) && (row_idx % scoring_batch_size == 0)) {
        if (scoring_threads.size() > 0) {
          scoring_threads[scoring_threads.size()-1].join();
        }
        thread t(print_scores, row_idx+1, ref(X), ref(feature_names),
                     ref(h), DENSITY, density_constant,
                     ref(deltamax), ref(shift),
                     cmsketches, // copy
                     ref(fs), mean_bincount, // copy
                     ref(anomalyscores), score_once,
                     npoints);
        scoring_threads.push_back(move(t));
      }
    }
    end = chrono::steady_clock::now();
    diff = chrono::duration_cast<chrono::milliseconds>(end - start);
    cerr << "done in " << diff.count() << "ms" << endl;

    cerr << "Waiting for lagging scoring threads... ";
    if (scoring_threads.size() > 0)
      scoring_threads[scoring_threads.size()-1].join();
    cerr << "done." << endl;

    // score tuples at the end
    cerr << "scoring " << nrows << " tuples... ";
    start = chrono::steady_clock::now();
    print_scores(nrows, ref(X), ref(feature_names),
                 ref(h), DENSITY, density_constant,
                 ref(deltamax), ref(shift),
                 cmsketches, // copy
                 ref(fs), mean_bincount, // copy
                 ref(anomalyscores), score_once,
                 npoints);
    end = chrono::steady_clock::now();
    diff = chrono::duration_cast<chrono::milliseconds>(end - start);
    cerr << "done in " << diff.count() << "ms" << endl;

    // done

    // debug: check distance approximation
    /*vector<float> true_distances;
    vector<float> approx_distances;
    for (uint i = 0; i < init_sample_size; i++) {
      for (uint j = 0; j < init_sample_size; j++) {
        true_distances.push_back(euclidean_distance(X[i], X[j]));
        approx_distances.push_back(euclidean_distance(Xpsample[i], Xpsample[j]));
      }
    }
    for (uint i = 0; i < true_distances.size(); i++)
      cout << setprecision(6) << true_distances[i] << " " << approx_distances[i] << endl;
    */

    // debug: print bincounts at each depth
    /*for (uint row_idx = 0; row_idx < nrows; row_idx++) {
      for (auto b : bincounts[row_idx]) {
        cout << setprecision(12) << b << " ";
      }
      cout << endl;
    }
    */

    // debug: verify mean bincounts
    /*for (uint i = 0; i < d; i++) {
      float m = 0.0;
      for (uint j = 0; j < nrows; j++) {
        m += bincounts[j][i];
      }
      m /= nrows;
      float t = 0.0;
      for (uint j = 0; j < c; j++) {
        t += mean_bincount[j][i];
      }
      t /= c;
      cout << m << " " << t/npoints << endl;
    }*/

    // debug: print lociscores at each depth
    /*for (uint row_idx = 0; row_idx < nrows; row_idx++) {
      for (auto b : lociscores[row_idx]) {
        cout << setprecision(12) << b << " ";
      }
      cout << endl;
    }*/

    // debug: print anomalyscores
    /*for (uint row_idx = 0; row_idx < nrows; row_idx++) {
      cout << setprecision(12) << anomalyscores[row_idx] << " ";
    }
    cout << endl;
    */

  } else {
    // unknown feature space
  }

  return 0;
}

void print_scores(uint row_idx, vector<vector<float>>& X, vector<string>& feature_names,
                  vector<uint64_t>& h, float density, float density_constant,
                  vector<vector<float>>& deltamax, vector<vector<float>>& shift,
                  vector<vector<unordered_map<vector<int>,int>>> cmsketches, // copy
                  vector<vector<uint>>& fs, vector<vector<float>> mean_bincount, // copy
                  vector<float>& anomalyscores, bool score_once,
                  float npoints) {
  cerr << "\tscoring at tuple: " << row_idx;
  if (score_once)
    cerr << " (score once)";
  cerr << endl;
  cout << row_idx << "\t";
  for (uint row_idx2 = 0; row_idx2 < row_idx; row_idx2++) {
    float anomalyscore;
    if (score_once) {
      anomalyscore = anomalyscores[row_idx2];
    } else {
      tie(anomalyscore, npoints) =
        chains_add(X[row_idx2], feature_names, h, density, density_constant, deltamax, shift,
                   cmsketches, fs, mean_bincount, npoints, false);
    }
    cout << setprecision(12) << anomalyscore << " ";
  }
  cout << endl;
}
