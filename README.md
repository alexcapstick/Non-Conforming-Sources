# Non Conforming Sources

This is the code made available for reproducing the results discussed in the ICLR 2023 Tiny Paper titled "Predicting Targets With Data from Non-Conforming Sources".


## What do I need to install?
The main requirements for running this code are:

```
python
jax
flax
torch
torchvision
wfdb
numpy
pandas
tqdm
imblearn
sklearn
```

## Where are the Models?

All of the models are implemented in JAX and can be found in the file `./experiment_code/model.py`. These are designed to be flexible and so the ResNet1D and MLP models contained in this file can be used with any dataset.

## Where are the Training and Testing Scripts?

All of the scripts used to run the experiments are located in the directory `./experiment_scripts/` which contains scripts for evaluating the MLP models on the MNIST dataset and the ResNet models on the PTB-XL dataset. Within this directory is also a bash script titled `run_experiments.txt` which contains the bash commands to run the experiments presented in the work.

## Where are the Results Saved?

After running the experiments, all files containing the results will be saved as pickled pandas files in the directory `./results/`, and all tensorboard runs will be saved in the directory `./tb_runs/scripts/`.