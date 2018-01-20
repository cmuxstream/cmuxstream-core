#include <cassert>
#include "chain.h"
#include <chrono>
#include "docopt.h"
#include "io.h"
#include <iostream>
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
              [--nfeatures=<number of features>]
              [--nwindows=<number of windows>]

      xstream (-h | --help)

    Options:
      -h, --help                        Show this screen.
      --input=<input file>              Input file.
      --k=<projection size>             Projection size.
      --c=<number of chains>            Number of chains.
      --d=<depth>                       Depth.   
      --nfeatures=<number of features>  Number of features, if fixed [default: -1].
      --nwindows=<number of windows>    Number of windows [default: 0].
)";


void process_unknown(int);
void process_fixed(int);

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
  chrono::nanoseconds diff; 

  // store argumetns
  string input_file(args["--input"].asString());
  uint k = args["--k"].asLong();
  uint c = args["--c"].asLong();
  uint d = args["--d"].asLong();
  
  int nfeatures = -1; // non-fixed feature space by default
  if (args.find("--nfeatures") != args.end()) {
    nfeatures = args["--nfeatures"].asLong();
  }
  
  int nwindows = 0; // no windows by default
  if (args.find("--nwindows") != args.end()) {
    nwindows = args["--nwindows"].asLong();
  }

  cerr << "xstream: "
       << "K=" << k << " C=" << c << " d=" << d << endl;
  cerr << "\tinput: " << input_file << endl;
  cerr << "\tno. of features: " << nfeatures << endl;
  cerr << "\tno. of windows: " << nwindows << endl;

  // initialize chains
  cerr << "initializing... ";
  start = chrono::steady_clock::now();

  vector<vector<float>> deltamax(c, vector<float>(k, 0.0));
  vector<vector<float>> shift(c, vector<float>(k, 0.0));
  vector<vector<unordered_map<string,int>>> cmsketches(c, vector<unordered_map<string,int>>(d));
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
  diff = chrono::duration_cast<chrono::nanoseconds>(end - start);
  cerr << "done in " << diff.count() << "ns" << endl;

  // input stream
  if (nfeatures > 0) {
    // fixed feature space

    // read input
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
    vector<int> bincounts(nrows);
    vector<float> lociscores(nrows);
    vector<float> anomalyscores(nrows);
    
    // set deltamax using an initial sample
    //chains_init_deltamax(X, INIT_SAMPLE_SIZE, deltamax, shift, prng);

    // stream in edges
    //cerr << "Streaming in " << num_test_edges << " tuples:" << endl;
    // LOOP
    //  process edge
    //    update chains
    //    update scores
  } else {
    // unknown feature space
  }

  return 0;
}
