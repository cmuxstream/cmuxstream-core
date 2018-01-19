#include <cassert>
#include <chrono>
#include "docopt.h"
#include "io.h"
#include <iostream>
#include <string>
#include <tuple>
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
  // arguments
  map<string, docopt::value> args = docopt::docopt(USAGE, { argv + 1, argv + argc });

  // for timing
  chrono::time_point<chrono::steady_clock> start;
  chrono::time_point<chrono::steady_clock> end;
  chrono::nanoseconds diff; 

  // store argumetns
  string input_file(args["--input"].asString());
  uint32_t k = args["--k"].asLong();
  uint32_t c = args["--c"].asLong();
  uint32_t d = args["--d"].asLong();
  
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

  // construct chains
    
  if (nfeatures > 0) {
    // fixed feature space
    
    // read input
    vector<vector<float>> X;
    vector<bool> Y;
    tie(X, Y) = input_fixed(input_file);

#ifdef VERBOSE
    for (uint i = 0; i < X.size(); i++) {
      vector<float> x = X[i];
      for (const auto& v: x) {
        cout << v << ",";
      }
      cout << "\t" << Y[i] << endl;
    }
#endif 

    // construct auxiliary data structures

    // set deltamax using a sample

    // stream in edges
    //cerr << "Streaming in " << num_test_edges << " tuples:" << endl;
    // LOOP
    //  process edge
    //    update chains
    //    update scores
    process_fixed(nwindows);
  } else {
    // unknown feature space
    process_unknown(nwindows);
  }

  return 0;
}

void process_fixed(int nwindows) {
}

void process_unknown(int nwindows) {
}
