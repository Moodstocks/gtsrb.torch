### Benchmark utility module

The [`bench_utils.lua`](../bench_utils.lua) module contains a useful function used by the benchmarking scripts.

#### Run the tests
`bench_utils.run_test(params, output)`

This function runs all the tests specified by `params` and print the results in `output`.
* `output` should be a .md file descriptor since the output will be written as a markdown table.
* `params` is a list of lua tables each containing two elements:
    * `1` contains the command line option
    * `2` contains all the values to test for this argument. It can take 3 types of values:
        * `true` will print the command line option with no value
        * `false` will print nothing relative to this option
        * other will print the command line option followed by the value

The benchmark will test all possible combination of values for all the options.

### Baseline benchmark
[`baseline.lua`](../baseline.lua)

This script launches all the tests relative to the basic nets on gtsrb.
It will outputs its results to [`baseline.md`](../baseline.md).


### Spatial tranformer benchmark
[`baseline_st.lua`](../baseline_st.lua)

This script launches all the tests relative to the spatial transformer networks on the gtsrb dataset.
It will output its results to [`baseline_st.md`](../baseline_st.md)


### Results

All the benchmarks are ran using only half the dataset (20 000 training samples) for 10 epochs to speedup the processing. The main baseline take 5 hours to run on a Titan X, the spatial transformer one take 48 hours on a Titan X (more than 150 different configurations tested).

Here are some insights we got from these benchmarks:

* Using more data always leads to better performances.
* The local and contrastive normalization are important on this dataset (very noisy initial images)
* With our benchmark parameters, adding momentum or multi-scale does not lead to improvement
* Use of bigger networks (depth or shape) leads to better performances (up to a certain size)
* In all tests, adding a spatial transformer lead to an improvement in performances
* As for the network, the bigger the localization network the better (up to a certain size)
* For small localization network, restraining the possible transformation for the spatial transformer lead to improvement (some exceptions can be found)
* Adding more than one st leads to mixed results (clear gain in accuracy in certain setup)
