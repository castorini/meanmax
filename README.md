# MeanMax
Implementations of the unbiased estimator and experiments from [Showing Your Work Doesn't Always Work](https://arxiv.org/abs/2004.13705) (ACL 2020).

## Citation
```
@misc{tang2020showing,
    title={Showing Your Work Doesn't Always Work},
    author={Raphael Tang and Jaejun Lee and Ji Xin and Xinyu Liu and Yaoliang Yu and Jimmy Lin},
    year={2020},
    eprint={2004.13705},
    archivePrefix={arXiv}
}
```

## Installation

1. Clone the repository: `git clone https://github.com/castorini/meanmax && cd meanmax`

2. With Python 3.7, install the requirements: `pip install -r requirements.txt` (use `virtualenv` if you want)

3. That's all!

## Experiments

### Quick demonstration

For a quick demonstration, you can use a reduced number of iterations at the cost of some precision.
However, it'll be sufficient to see the effects of bias and ill-constructed confidence intervals. 

#### Drawing MeanMax curves
To draw biased MeanMax curves, run 
```bash
python -m meanmax.run.simulate --action mme -k 15 -n 15 -n 2000 --mc-total 10000
```

To draw **unbiased** MeanMax curves, run
```bash
python -m meanmax.run.simulate --action mme -k 15 -n 15 -n 2000 --mc-total 10000 --unbiased
```

The unbiased curve should be closer to the true curve.

#### False conclusion probing
To see the proportion of negative errors for the biased estimator, run
```bash
python -m meanmax.run.simulate --action "mme-test" -s 30 --start-no 25 -n 5000
```

For the unbiased estimator, run
```bash
python -m meanmax.run.simulate --action "mme-test" -s 30 --start-no 25 -n 5000 --unbiased
```

The first should be around 68 and the second around 50.

#### CI coverage test
To see the empirical coverage using the percentile bootstrap, run
```bash
python -m meanmax.run.simulate --action bs -s 30 --start-no 25 -n 100 --mc-total 1000
```

### LSTM and MLP

For convenience, first run
```bash
alias process_hedwig="(tail -n +2 data/hedwig.tsv | grep reg_lstm | cut -d$'\t' -f5 && echo && tail -n +2 data/hedwig.tsv | grep mlp | cut -d$'\t' -f5)"
```
Then, for each of the following scripts, append the `--unbiased` option to use the unbiased estimator, and `--swapped` to use the MLP instead of the LSTM.  

**Drawing MeanMax curves**: `process_hedwig | python -m meanmax.run.simulate --action mme --use-kde`

**False conclusion probing**: `process_hedwig | python -m meanmax.run.simulate --action mme-test --use-kde`

**CI coverage**: `process_hedwig | python -m meanmax.run.simulate --action bs --use-kde -k <the k to test>`

