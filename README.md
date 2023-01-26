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

### Data loading
Depending on the name of the file, the required dataset is called. In the sample code we use a relative path, placing the dataset and the code in a folder. After processing the dataset, the data is split using train_test_split from the sklearn package. 20% of the data is used as the final test set and 80% of the data is used as the training set.
```
X_train_general, X_test_general, y_train_general, y_test_general= train_test_split(x_general_1, y_general,test_size=0.2)
```
### Using optuna to get the best model parameters
The training data set is divided into ten parts using cross-validation, and ten calculations are performed to find the average of the accuracies. The parameters of the model are tuned using optuna, and the cross-validation process is repeated with different models to derive good model parameters.
### Data analysis
Using the best model parameters obtained in the second step, a new model is generated, which is used to learn the training set and subsequently validated using the test set to obtain the final results
