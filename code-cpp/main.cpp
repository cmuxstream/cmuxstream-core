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
              [--fixed]
              [--nwindows=<number of windows>]
              [--initsample=<initial sample size>]
              [--scoringbatch=<scoring batch size>]
              [--score-once]
              [--no-project]

      xstream (-h | --help)

    Options:
      -h, --help                           Show this screen.
      --k=<projection size>                Projection size [default: 100].
      --c=<number of chains>               Number of chains [default: 100].
      --d=<depth>                          Depth [default: 15].
      --fixed                              Fixed feature space.
      --nwindows=<number of windows>       Number of windows [default: 1].
      --initsample=<initial sample size>   Initial sample size [default: 256].
      --scoringbatch=<scoring batch size>  Scoring batch size [default: 1000].
      --score-once                         Score each point only once.
      --no-project                         Do not project input tuples.
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

  // store argumetns
  uint k = args["--k"].asLong();
  uint c = args["--c"].asLong();
  uint d = args["--d"].asLong();
  bool fixed = args["--fixed"].asBool();
  bool score_once = args["--score-once"].asBool();
  bool no_project = args["--no-project"].asBool();

  int nwindows = args["--nwindows"].asLong(); // no windows by default
  uint init_sample_size = args["--initsample"].asLong(); // full data size by default
  uint scoring_batch_size = args["--scoringbatch"].asLong(); // 1000 by default

  cerr << "xstream: " << "K=" << k << " C=" << c << " d=" << d << " ";
  if (fixed) cerr << "(fixed feature space)";
  if (nwindows > 0) cerr << " (windowed)";
  if (score_once) cerr << " (score-once)";
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

  // only required for retrospective scoring, not part of the model
  vector<vector<float>> alltuples;
  alltuples.reserve(1000000 * 100);

  // only required for score-once scoring
  vector<float> anomalyscores;
  anomalyscores.reserve(1000000);

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

    vector<string> fields;
    string field;
    while (ss >> field) { fields.push_back(field); }
    fields.pop_back(); // delete label

    vector<float> xp;
    if (no_project) {
      // do not project tuple
      if (!fixed) {
        cerr << "No projection only possible with fixed feature spaces!" << endl;
        exit(1);
      }
      if (k != fields.size()) {
        cerr << "k = " << k << ", D = " << fields.size() << ": ";
        cerr << "No projection requires k be equal to the dimensionality!" << endl;
        exit(1);
      }
      for (uint j = 0; j < fields.size(); j++) {
        xp.push_back(atof(fields[j].c_str()));
      }
    } else {
      // project tuple
      xp = streamhash_project(fields, fixed, h, DENSITY, density_constant);
    }
    alltuples.push_back(xp);

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

      // compute deltmax/shift from initial sample
      cerr << "initializing deltamax from sample size " << window_size << "..." << endl;
      tie(deltamax, shift) = compute_deltamax(window, c, k, prng);

      // add initial sample tuples to chains
      for (auto x : window) { chains_add(x, deltamax, shift, cmsketches, fs, true); }

      // score initial sample tuples
      cerr << "scoring first batch of " << init_sample_size << " tuples... " << endl;
      for (auto x : window) {
        float anomalyscore;
        //vector<float> bincount;
        //tie(anomalyscore, bincount) = chains_add(x, deltamax, shift, cmsketches, fs, false);
        anomalyscore = chains_add(x, deltamax, shift, cmsketches, fs, false);
        anomalyscores.push_back(anomalyscore);
      }

      window_size = 0;
      row_idx++;
      continue;
    }

    // row_idx > init_sample_size

    if (nwindows <= 0) { // non-windowed mode

      // add point to chains, store anomaly scores for score-once mode
      float anomalyscore;
      //vector<float> bincount;
      //tie(anomalyscore, bincount) = chains_add(xp, deltamax, shift, cmsketches, fs, true);
      anomalyscore = chains_add(xp, deltamax, shift, cmsketches, fs, true);
      anomalyscores.push_back(anomalyscore);

    } else if (nwindows > 0) { // windowed mode
      window[window_size] = xp;
      window_size++;

      // store anomaly-score for score-once mode, do not update chains
      if (score_once) {
        float anomalyscore;
        //vector<float> bincount;
        //tie(anomalyscore, bincount) = chains_add(xp, deltamax, shift, cmsketches, fs, false);
        anomalyscore = chains_add(xp, deltamax, shift, cmsketches, fs, false);
        anomalyscores.push_back(anomalyscore);
      }

      // if the batch limit is reached, construct new chains
      if (window_size == static_cast<uint>(init_sample_size)) {
        cerr << "\tnew chains at tuple: " << row_idx << endl;

        // new deltamax, shift from the cached points
        tie(deltamax, shift) = compute_deltamax(window, c, k, prng);

        // clear old bincounts
        for (uint chain = 0; chain < c; chain++) {
          for (uint depth = 0; depth < d; depth++) {
            cmsketches[chain][depth].clear();
          }
        }

        // add current window tuples to chains
        for (auto x : window) { chains_add(x, deltamax, shift, cmsketches, fs, true); }

        window_size = 0;
      }
    }

    if (!score_once) {
      // retrospective scoring
      if ((row_idx > init_sample_size) && (row_idx % scoring_batch_size == 0)) {
        cerr << "\tscoring at tuple: " << row_idx << endl;
        cout << row_idx << "\t";
        for (uint i = 0; i < alltuples.size(); i++) {
          float anomalyscore;
          vector<float> bincount;
          cout << setprecision(12) << anomalyscore << " ";
        }
        cout << endl;
      }
    }

    row_idx++;
  }
  end = chrono::steady_clock::now();
  diff = chrono::duration_cast<chrono::milliseconds>(end - start);
  cerr << "done in " << diff.count() << "ms" << endl;

  if (init_sample_size > alltuples.size()) {
    cerr << "Initial sample > no. of tuples: no chains constructed!" << endl;
    exit(1);
  }

  // score tuples at the end
  cerr << "final scoring of " << alltuples.size() << " tuples... ";
  start = chrono::steady_clock::now();
  cout << alltuples.size() << "\t";
  for (uint i = 0; i < alltuples.size(); i++) {
    float anomalyscore;
    if (score_once) {
      anomalyscore = anomalyscores[i];
    } else {
      //vector<float> bincounts;
      //tie(anomalyscore, bincounts) = chains_add(alltuples[i], deltamax, shift, cmsketches, fs,
      //                                          false);
      anomalyscore = chains_add(alltuples[i], deltamax, shift, cmsketches, fs, false);
    }
    cout << setprecision(12) << anomalyscore << " ";
  }
  cout << endl;
  end = chrono::steady_clock::now();
  diff = chrono::duration_cast<chrono::milliseconds>(end - start);
  cerr << "done in " << diff.count() << "ms" << endl;

  // print bincounts
  /*
  cerr << "bincounts..." << endl;
  for (uint i = 0; i < alltuples.size(); i++) {
    float anomalyscore;
    vector<float> bincount;
    tie(anomalyscore, bincount) = chains_add(alltuples[i], deltamax, shift, cmsketches, fs, false);
    for (uint j = 0; j < d-1; j++) {
      cout << setprecision(12) << bincount[j] << " ";
    }
    cout << setprecision(12) << bincount[d-1] << endl;
  }
  */

  return 0;
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
      if (abs(deltamax[i][j]) <= EPSILON) deltamax[i][j] = 1.0;
      uniform_real_distribution<> dis(0, deltamax[i][j]);
      shift[i][j] = dis(prng);
    }
  }

  return make_tuple(deltamax, shift);
}
