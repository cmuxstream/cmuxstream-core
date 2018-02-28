# xStream - C++

[https://cmuxstream.github.io](https://cmuxstream.github.io)

This implementation is in C++ 11 and works on static, row-stream and evolving-stream data.

## Format

Static and row-stream data needs to be in the SVM-Light format (see `synDataNoisy.svm`, the
SVM-Light formatted version of `synDataNoisy.tsv` used to test the
[Python implementation](https://github.com/cmuxstream/cmuxstream-core/tree/master/python)).
Evolving-stream data needs to be in the modified SVM-Light format described in the
[datasets README](https://github.com/cmuxstream/cmuxstream-data/tree/master/evolving).

## Build

```
make clean
make optimized
```

## Arguments and Examples

Help with arguments can be displayed by running `./xstream --help`
```
    Usage:
      xstream [--k=<projection size>]
              [--c=<number of chains>]
              [--d=<depth>]
              [--rowstream]
              [--nwindows=<windowed>]
              [--initsample=<initial sample size>]
              [--scoringbatch=<scoring batch size>]
              [--cosine]

      xstream (-h | --help)

    Options:
      -h, --help                           Show this screen.
      --k=<projection size>                Projection size [default: 100].
      --c=<number of chains>               Number of chains [default: 100].
      --d=<depth>                          Depth [default: 15].
      --rowstream                          Row stream (each row starts with a label).
      --nwindows=<windowed>                > 0 if windowed [default: 1].
      --initsample=<initial sample size>   Initial sample size [default: 256].
      --scoringbatch=<scoring batch size>  Print scores at regular intervals [default: 1000].
      --cosine                             Work in cosine space instead of Euclidean.
```

If the data is static or a row-stream in SVM-Light format, specify the `--rowstream` option.

An example of running on 3082 rows of the [synthetic data](https://github.com/cmuxstream/cmuxstream-core/tree/master/python)
without windows (mimics the static Python implementation) in Euclidean space, scoring just once at the end:
```
cat synDataNoisy.svm | ./xstream --k 50 --c 50 --d 10 --rowstream --nwindows 0 --initsample `wc -l < synDataNoisy.svm` --scoringbatch 100000 > scores
python test_xstream_static.py
```

An example of evaluating these scores is in `test_xstream_static.py`.

# Contact

   * emaad@cmu.edu
   * hlamba@andrew.cmu.edu
   * lakoglu@andrew.cmu.edu
