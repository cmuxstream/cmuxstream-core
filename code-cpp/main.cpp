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

void
print_scores(uint row_idx, vector<vector<float>>& X, vector<string>& feature_names,
             vector<uint64_t>& h, float density, float density_constant,
             vector<vector<float>>& deltamax, vector<vector<float>>& shift,
             vector<vector<unordered_map<vector<int>,int>>>& cmsketches, // copy
             vector<vector<uint>>& fs,
             vector<float>& anomalyscores, bool score_once);

tuple<vector<vector<float>>,vector<vector<float>>>
compute_deltamax(vector<uint>& point_cache, vector<vector<float>>& X,
                 vector<string>& feature_names, vector<uint64_t>& h, float density,
                 float density_constant, uint c, uint k, mt19937_64& prng);

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

  cerr << "xstream: " << "K=" << k << " C=" << c << " d=" << d << " ";
  if (fixed) cerr << "(fixed feature space)";
  if (nwindows > 0) cerr << " (windowed)";
  if (score_once) cerr << " (score-once)";
  cerr << endl;
  cerr << "\tinput: " << input_file << endl;
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
    cerr << "reading input tuples..." << endl;
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

    // auxilliary date structures
    vector<float> anomalyscores(nrows); // needed in the score-once mode
    vector<uint> point_cache; // needed in the windowed-mode

    // construct feature names
    vector<string> feature_names(ndims);
    for (uint i = 0; i < ndims; i++) {
      feature_names[i] = to_string(i);
    }

    if (init_sample_size < 0) {
      init_sample_size = nrows; // = \phi if nwindows > 0
    } else {
      if (static_cast<uint>(init_sample_size) > nrows) {
        cerr << "ERROR: init sample size > nrows" << endl;
        exit(-1);
      }
    }

    // construct projection of an initial sample, compute projection range
    cerr << "initializing deltamax from sample size " << init_sample_size << "..." << endl;
    vector<uint> init_sample_points(init_sample_size);
    for (uint i = 0; i < static_cast<uint>(init_sample_size); i++) {
      init_sample_points[i] = i;
    }
    tie(deltamax, shift) = compute_deltamax(init_sample_points, X, feature_names, h, DENSITY,
                                            density_constant, c, k, prng);

    // stream in the initial batch of tuples: identical for windowed/non-windowed modes
    cerr << "streaming in first batch of " << init_sample_size << " tuples... " << endl;
    start = chrono::steady_clock::now();
    for (uint row_idx = 0; row_idx < static_cast<uint>(init_sample_size); row_idx++) {
      chains_add(X[row_idx], feature_names, h, DENSITY, density_constant,
                 deltamax, shift, cmsketches, fs, true);

    }
    end = chrono::steady_clock::now();
    diff = chrono::duration_cast<chrono::milliseconds>(end - start);
    cerr << "done in " << diff.count() << "ms" << endl;

    // re-score first batch of tuples using the entire training sample
    cerr << "scoring first batch of " << init_sample_size << " tuples... " << endl;
    start = chrono::steady_clock::now();
    for (uint row_idx = 0; row_idx < static_cast<uint>(init_sample_size); row_idx++) {
      if (score_once) {
        anomalyscores[row_idx] = chains_add(X[row_idx], feature_names, h, DENSITY, density_constant,
                                            deltamax, shift, cmsketches, fs, false);
      }

      // print scores at regular intervals
      if ((row_idx > 0) && ((row_idx+1) % scoring_batch_size == 0)) {
        print_scores(row_idx+1, ref(X), ref(feature_names),
                     ref(h), DENSITY, density_constant,
                     ref(deltamax), ref(shift),
                     cmsketches, // copy
                     ref(fs),
                     ref(anomalyscores), score_once);
      }
    }
    end = chrono::steady_clock::now();
    diff = chrono::duration_cast<chrono::milliseconds>(end - start);
    cerr << "done in " << diff.count() << "ms" << endl;

    // stream in remaining tuples
    cerr << "streaming in remaining " << nrows - init_sample_size << " tuples..." << endl;
    start = chrono::steady_clock::now();
    for (uint row_idx = init_sample_size; row_idx < nrows; row_idx++) {
      if (nwindows <= 0) {
        // non-windowed mode

        // add point to chains
        float anomalyscore = chains_add(X[row_idx], feature_names, h, DENSITY, density_constant,
                                        deltamax, shift, cmsketches, fs, true);

        // store anomaly-score for score-once mode
        if (score_once) { anomalyscores[row_idx] = anomalyscore; }

        // print progress
      } else if (nwindows > 0) {
        // windowed mode

        // add point to cache
        point_cache.push_back(row_idx);

        // store anomaly-score for score-once mode, do not update chains
        if (score_once) {
          float anomalyscore = chains_add(X[row_idx], feature_names, h, DENSITY, density_constant,
                                          deltamax, shift, cmsketches, fs, false);
          anomalyscores[row_idx] = anomalyscore;
        }

        // if the cache limit is reached, construct new chains
        if (point_cache.size() == static_cast<uint>(init_sample_size)) {
          //cerr << "\tnew chains at tuple: " << row_idx + 1 << endl;

          // new deltamax, shift from the cached points
          tie(deltamax, shift) = compute_deltamax(point_cache, X, feature_names, h, DENSITY,
                                                  density_constant, c, k, prng);

          // clear old bincounts
          for (uint chain = 0; chain < c; chain++) {
            for (uint depth = 0; depth < d; depth++) {
              cmsketches[chain][depth].clear();
            }
          }

          // add cached points to chains
          for (auto pidx : point_cache) {
            chains_add(X[pidx], feature_names, h, DENSITY, density_constant,
                       deltamax, shift, cmsketches, fs, true);
          }

          // clear point cache
          point_cache.clear();
        }
      }

      // print scores for all seen points at regular intervals
      if ((row_idx > 0) && ((row_idx+1) % scoring_batch_size == 0)) {
        print_scores(row_idx+1, ref(X), ref(feature_names),
                     ref(h), DENSITY, density_constant,
                     ref(deltamax), ref(shift),
                     cmsketches, // copy
                     ref(fs),
                     ref(anomalyscores), score_once);
      }
    }
    end = chrono::steady_clock::now();
    diff = chrono::duration_cast<chrono::milliseconds>(end - start);
    cerr << "done in " << diff.count() << "ms" << endl;

    // score tuples at the end
    cerr << "final scoring..." << endl;
    start = chrono::steady_clock::now();
    print_scores(nrows, ref(X), ref(feature_names),
                 ref(h), DENSITY, density_constant,
                 ref(deltamax), ref(shift),
                 cmsketches, // copy
                 ref(fs),
                 ref(anomalyscores), score_once);
    end = chrono::steady_clock::now();
    diff = chrono::duration_cast<chrono::milliseconds>(end - start);
    cerr << "done in " << diff.count() << "ms" << endl;

    // done
  } else {
    // unknown feature space
  }

  return 0;
}

tuple<vector<vector<float>>,vector<vector<float>>>
compute_deltamax(vector<uint>& point_cache, vector<vector<float>>& X,
                 vector<string>& feature_names, vector<uint64_t>& h, float density,
                 float density_constant, uint c, uint k, mt19937_64& prng) {

  vector<vector<float>> deltamax(c, vector<float>(k, 0.0));
  vector<vector<float>> shift(c, vector<float>(k, 0.0));

  vector<float> dim_min(k, numeric_limits<float>::max());
  vector<float> dim_max(k, numeric_limits<float>::min());

  for (auto pidx : point_cache) {
    vector<float> Xp = streamhash_project(X[pidx], feature_names, h, DENSITY,
                                          density_constant);
    for (uint j = 0; j < k; j++) {
      if (Xp[j] > dim_max[j]) { dim_max[j] = Xp[j]; }
      if (Xp[j] < dim_min[j]) { dim_min[j] = Xp[j]; }
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

void print_scores(uint row_idx, vector<vector<float>>& X, vector<string>& feature_names,
                  vector<uint64_t>& h, float density, float density_constant,
                  vector<vector<float>>& deltamax, vector<vector<float>>& shift,
                  vector<vector<unordered_map<vector<int>,int>>>& cmsketches, // copy
                  vector<vector<uint>>& fs,
                  vector<float>& anomalyscores, bool score_once) {
  cerr << "\tscoring at tuple: " << row_idx << endl;
  cout << row_idx << "\t";
  for (uint row_idx2 = 0; row_idx2 < row_idx; row_idx2++) {
    float anomalyscore;
    //uint depth = cmsketches[0].size();
    //vector<float> avg_bincount(depth), avg_lociscore(depth);
    if (score_once) {
      anomalyscore = anomalyscores[row_idx2];
    } else {
      //tie(avg_bincount, avg_lociscore, anomalyscore, npoints) =
      anomalyscore = chains_add(X[row_idx2], feature_names, h, density, density_constant,
                                deltamax, shift, cmsketches, fs, false);
    }
    cout << setprecision(12) << anomalyscore << " ";
    /*for (uint d = 0; d < depth-1; d++)
      cout << setprecision(12) << avg_bincount[d] << " ";
    cout << setprecision(12) << avg_bincount[depth-1] << "\t";
    for (uint d = 0; d < depth-1; d++)
      cout << setprecision(12) << avg_lociscore[d] << " ";
    cout << setprecision(12) << avg_lociscore[depth-1] << endl;*/
  }
  cout << endl;
}
