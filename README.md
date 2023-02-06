# BDE-binge-drinking

# Introduction

The Code folder holds our program code and the dataset file holds the datasets we used for the analysis with different windows and different distances. When running the program, to prevent path configuration problems, we recommend placing the dataset and the code in the same folder. To ensure the privacy of the participants, we have hidden some of the location data, so the results obtained using this dataset may be different from the results in the paper.

# Dataset
We use behavioral features from sensor data over a given analysis window of w hours, we predict whether the subject will have a binge drinking event within d hours . To build this predictive model, we first created a dataset containing the sensor data for each person on each day of the study, calculating features in 15-minute windows.

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
The training data set is divided into ten parts using cross-validation, and ten calculations are performed to find the average of the accuracies. 
```
for train_index, test_index in kfold.split(x_general_gk):
  sub_x_train_general, sub_x_test_general = x_general_gk[train_index], x_general_gk[test_index]
  sub_y_train_general, sub_y_test_general = y_general_gk[train_index], y_general_gk[test_index]
```
Since each class has a different amount of data, we use smote to average the data
```
sub_x_train_general, sub_y_train_general = sm.fit_resample(sub_x_train_general, sub_y_train_general)
```
The parameters of the model are tuned using optuna, and the cross-validation process is repeated with different models to derive good model parameters.

optunan website: https://optuna.org/
### Data analysis
Using the best model parameters obtained in the second step, a new model is generated, which is used to learn the training set and subsequently validated using the test set to obtain the final results
```
model = XGBClassifier(**params)
```

### XAI ANALYSIS
## SETUP 
# Partial Dependence Plot- pdpbox package Installation

through pip 
```
$ pip install pdpbox
```
through git
```
$ git clone https://github.com/SauceCat/PDPbox.git
$ cd PDPbox
$ python setup.py install
```
For details see:
https://github.com/SauceCat/PDPbox
