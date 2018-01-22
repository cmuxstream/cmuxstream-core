#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <tuple>
#include <vector>

namespace std {

tuple<vector<vector<float>>, vector<bool>>
  input_fixed(string filename) {

    vector<vector<float>> X;
    vector<bool> Y;

    ifstream f(filename);
    stringstream ss;
    string line;

    while (getline(f, line)) {
      ss.clear();
      stringstream ss(line);

      vector<float> xy;
      float val;
      while (ss >> val) {
        xy.push_back(val);
      }

      vector<float> x(xy.begin(), xy.end() - 1); // data
      bool y = static_cast<bool>(xy[xy.size()-1]); // label
      X.push_back(x);
      Y.push_back(y); // label
    }

    f.close();

    return make_tuple(X, Y);
  }

}
