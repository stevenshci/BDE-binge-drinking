# HDE-binge-drinking

# Introduction

The Code folder holds our program code and the dataset file holds the datasets we used for the analysis with different windows and different distances. When running the program, to prevent path configuration problems, we recommend placing the dataset and the code in the same folder.

# Dataset


# Tutorial

## Setup

### Environment

Here is an example of using anaconda or miniconda for environment setup:

```
conda create -n globem python=3.9
conda activate globem
pip install -r requirements.txt
```

## Code Breakdown
For the analysis of the data, the code can be understood as three parts
1. Data loading
2. Using optuna to get the best model parameters
3. Data analysis
