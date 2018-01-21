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

      xstream (-h | --help)

    Options:
      -h, --help                        Show this screen.
      --input=<input file>              Input file.
      --k=<projection size>             Projection size.
      --c=<number of chains>            Number of chains.
      --d=<depth>                       Depth.   
      --fixed                           Fixed feature space.
      --nwindows=<number of windows>    Number of windows [default: 0].
)";

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

  int nwindows = 0; // no windows by default
  if (args.find("--nwindows") != args.end()) {
    nwindows = args["--nwindows"].asLong();
  }

  cerr << "xstream: "
       << "K=" << k << " C=" << c << " d=" << d
       << " windows=" << nwindows << endl;
  cerr << "\tinput: " << input_file << " ";
  if (fixed) cerr << "(fixed feature space)";
  cerr << endl;

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
    vector<float> lociscores(nrows);
    vector<float> anomalyscores(nrows);

    // construct feature names
    vector<string> feature_names(ndims);
    for (uint i = 0; i < ndims; i++) {
      feature_names[i] = to_string(i);
    }

    // construct projection of an initial sample, compute projection range
    cerr << "Initializing deltamax from sample size " << INIT_SAMPLE_SIZE << "..." << endl;
    vector<vector<float>> Xpsample(INIT_SAMPLE_SIZE, vector<float>(k, 0.0));
    vector<float> dim_min(k, numeric_limits<float>::max());
    vector<float> dim_max(k, numeric_limits<float>::min());
    for (uint i = 0; i < INIT_SAMPLE_SIZE; i++) {
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
    cerr << "streaming in " << nrows << " tuples... ";
    start = chrono::steady_clock::now();
    for (uint row_idx = 0; row_idx < nrows; row_idx++) {
      vector<float> bincount;
      float lociscore, anomalyscore;
      tie(bincount, lociscore, anomalyscore) = chains_add(X[row_idx], feature_names, h, DENSITY,
                                                          density_constant, deltamax, shift,
                                                          cmsketches, fs, true);
      //lociscores[row_idx] = l;
      //anomalyscores[row_idx] = s;
    }
    end = chrono::steady_clock::now();
    diff = chrono::duration_cast<chrono::milliseconds>(end - start);
    cerr << "done in " << diff.count() << "ms" << endl;

    // score tuples
    cerr << "scoring " << nrows << " tuples... ";
    start = chrono::steady_clock::now();
    for (uint row_idx = 0; row_idx < nrows; row_idx++) {
      vector<float> bincount;
      float lociscore, anomalyscore;
      tie(bincount, lociscore, anomalyscore) = chains_add(X[row_idx], feature_names, h, DENSITY,
                                                          density_constant, deltamax, shift,
                                                          cmsketches, fs, false);
      bincounts[row_idx] = bincount;
      //lociscores[row_idx] = l;
      //anomalyscores[row_idx] = s;
    }
    end = chrono::steady_clock::now();
    diff = chrono::duration_cast<chrono::milliseconds>(end - start);
    cerr << "done in " << diff.count() << "ms" << endl;

    // done

    // debug: check distance approximation
    /*vector<float> true_distances;
    vector<float> approx_distances;
    for (uint i = 0; i < INIT_SAMPLE_SIZE; i++) {
      for (uint j = 0; j < INIT_SAMPLE_SIZE; j++) {
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
  } else {
    // unknown feature space
  }

  return 0;
}
