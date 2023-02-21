# Leveraging Mobile Phone Sensors, Machine Learning and Explainable Artificial Intelligence to Predict Imminent Same-Day Binge Drinking Events to Support Just-In-Time Adaptive Interventions: A Feasibility Study

# Introduction

This repository host the codebase and dataset used in the analysis of our work [Leveraging Mobile Phone Sensors, Machine Learning and Explainable Artificial Intelligence to Predict Imminent Same-Day Binge Drinking Events to Support Just-In-Time Adaptive Interventions: A Feasibility Study](https://example.com) with different windows and different distances from the drinking event. We evaluated the feasibility of using machine learning models to predict same-day BDEs (versus low-risk drinking events and non-drinking periods) using smartphone sensor data (e.g., accelerometer, GPS). Different algorithms (e.g., XGBoost, decision tree) were tested to assess the prediction accuracy across various “prediction distance” time windows (1-6 hours from drinking onset). Using a publicly available dataset, the best model was found to be a XGBoost model for both the weekend and weekday, with a prediction distance of 6 hours. Explainable AI (XAI) was used to explore interactions between the most informative phone sensor features contributing to BDEs.

## Highlights
SHAP Interaction between time of day and radius of gyration and average latitude features, respectively on weekdays (left) and on weekends (right) affecting the BDE prediction, using the test data set 
![image](/figs/rog_shap.png)


PDP Contour plot of the interaction between a 15-minute average latitude and longitude predicting BDEs on weekdays (left) and weekends (right) using the test data set
![image](/figs/contour_pdp.png)

# Dataset
Our strategy for drinking prediction modeling is as follows: using behavioral features from sensor data over a given analysis window of _w_ hours (where we varied _w_), we predicted whether the subject will have a binge drinking event within _d_ hours (with varying _d_ to optimize the model). We use behavioral features from sensor data over a given analysis window of _w_ hours, we predict whether the subject will have a binge drinking event within _d_ hours . To build this predictive model, we first created a dataset containing the sensor data for each person on each day of the study, calculating features in 15-minute windows. In the dataset folder, we have created 15 datasets by combining different windows and distances. The corresponding windows and diatances according to the name of the [dataset folder](/dataset/), such as the '15_result_indexed_dataset_window1_distance1.csv' file, window1 means its window is 1 and distance1 means its distance is 1. In total we have phone sensor data from 75 young adults.

![image](/figs/datasetCreation.png)

# Preprocessing, Modeling and Explanation

## Feature Extraction
We followed robust feature exctraction method decribed in [Extraction of Behavioral Features from Smartphone and Wearable Data](
https://doi.org/10.48550/arXiv.1812.10394). Additionaly we extarct [Latitude and Longitude](/code/Latitude%20and%20longitude%20extraction.py).

<!-- ## Machine Learning

## Model Explainbility -->

![image](/figs/ml_xai.png)

<!-- # Tutorial --> 

## Setup

### Environment

Here is an example of using anaconda or miniconda for environment setup:

For mac user:

```
conda create -n bde python=3.8
conda activate bde
pip install -r requirements.txt
```

For windows user:  
Please ensure that you are using python 3.8 environment.  
Please make sure to change the path of requirement.txt during installation.
```
pip install -r e:\\requirements.txt
```

If pdpbox install unsuccessful. Please through git (latest develop version):
```
$ git clone https://github.com/SauceCat/PDPbox.git
$ cd PDPbox
$ python setup.py install
```
### Running Code
```
conda activate bde
python Weekday_20%_optuna.py.py
python Weekend_20%_optuna.py.py
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

## XAI analysis
When we get the model, we use XAI to analysis it. For XAI analysis, we use two packages: pdpbox and shap.

For details see: https://github.com/slundberg/shap and https://github.com/SauceCat/PDPbox

### PDPBOX
We mainly use pdpbox to analyze PDP Contour, whose main code is shown below:
```
pdp_dist = pdp.pdp_isolate(model=model, dataset=X_test_general, model_features=X_test_general.columns, feature='radius_of_gyration', num_grid_points=11)
```


### SHAP
For SHAP, we used its analysis of the feature importance and Beeswarm Summary of the model.

The Beeswarm Summary code is:
```
print(shap.summary_plot(shap_values[2], X_test_general, max_display=20))
```
The code for Feature Importance is:
```
print(shap.summary_plot(shap_values[2], X_test_general, max_display=20, plot_type='bar'))
```
