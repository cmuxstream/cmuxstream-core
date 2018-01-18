#include <chrono>
#include <iostream>

#include "docopt.h"

using namespace std;

static const char USAGE[] =
R"(xstream.

    Usage:
      xstream --input=<input file>
              --k=<projection size>
              --c=<number of chains>
              --d=<depth> 

      xstream (-h | --help)

    Options:
      -h, --help                        Show this screen.
      --input=<input file>              Input file.
      --k=<projection size>             Projection size.
      --c=<number of chains>            Number of chains.
      --d=<depth>                       Depth.   
)";

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

  cout << "xstream: "
       << "K=" << k << " C=" << c << " d=" << d << endl;
  cout << "input: " << input_file << endl;

  // read input 

  // construct auxiliary data structures
  
  // set up universal hash family for StreamHash

  // construct chains

  // stream in edges
  //cout << "Streaming in " << num_test_edges << " tuples:" << endl;
  // LOOP
  //  process edge
  //    update chains
  //    update scores

  return 0;
}
