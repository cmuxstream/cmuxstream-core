# xStream - Python

[https://cmuxstream.github.io](https://cmuxstream.github.io)

This implementation is in Python 2.7 and works on static data.

## Prerequisites

Python packages can be installed via pip:

```
pip install -r requirements.txt
```

## Example

The `test_xstream.py` script contains an example of using xStream on a synthetic
static dataset `synDataNoisy.tsv`. The original dataset contained 3082 rows in
3 dimensions, with two non-anomalous clusters (depicted below in blue and orange)
and various anomalies.

<img src="https://raw.githubusercontent.com/cmuxstream/cmuxstream-core/master/python/synData.png" height="300" align="right"/>

The noisy dataset contains 100 additional columns containing Gaussian noise,
with mean 0.5 and standard deviation 0.05.

To test xStream and iForest on this noisy dataset:

```
python test_xstream.py
```

# Contact

   * emaad@cmu.edu
   * hlamba@andrew.cmu.edu
   * lakoglu@andrew.cmu.edu
