#ifndef XSTREAM_IO_H_
#define XSTREAM_IO_H_

#include <string>
#include <tuple>
#include <vector>

namespace std {

  tuple<vector<vector<float>>,vector<bool>>
    input_fixed(string filename);

}

#endif
