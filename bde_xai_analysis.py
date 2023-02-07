import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
import csv
import ast
from pdpbox.pdp import pdp_isolate, pdp_interact, pdp_interact_plot, pdp_plot
from pdpbox import pdp, get_dataset, info_plots

file_name = '15_result_indexed_dataset_window3_distance1.csv'
df=pd.read_csv(file_name)
df['WTSD_latitude'].fillna(0, inplace=True)
df['WTSD_longitude'].fillna(0, inplace=True)
df_omit=df.dropna()

#Select Weekdays
df_omit = df_omit[df_omit['day_of_week'] <5]

df_omit=df_omit.drop(columns='day_of_week')
df_omit["label"] = pd.factorize(df_omit["label"])[0].astype(int)
df_omit=df_omit.drop(columns=['pid','study_duration','Unnamed: 0','time_stamp'])

X = df_omit.drop(columns=['label'])
y = df_omit.label

#import the model parameter for the dataset
row = 2; col = 16 
with open("15_result_indexed_dataset_window3_distance1_weekday_result.csv") as f:
    reader = csv.reader(f)
    for i in range(row):
        row = next(reader)
    params = row[col-1]
params=ast.literal_eval(params)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model= XGBClassifier(**params).fit(X_train, y_train)
y_pred = model.predict(X_test)
score=accuracy_score(y_pred,y_test)
print('Accuracy: \n', score)

"""
You can select a smaller subset of X_test to analyze for quick results. 
"""
explainer = shap.KernelExplainer(model.predict_proba, X_test)
shap_values = explainer.shap_values(X_test)

#Beeswarm Summary Plot
print(shap.summary_plot(shap_values[2], X_test, max_display=20))
#Feature Importance Bar Plot
print(shap.summary_plot(shap_values[2], X_test, max_display=20, plot_type='bar'))

#SHAP Dependence Plot 
print(shap_figure=shap.dependence_plot('radius_of_gyration', shap_values[2], X_test, interaction_index='time_of_day'))

#Partial Dependence Plot (Radius of Gyration)
pdp_dist = pdp.pdp_isolate(model=model, dataset=X_test, model_features=X_test.columns, feature='radius_of_gyration', num_grid_points=11)
pdp.pdp_plot(pdp_dist, 'radius_of_gyration', x_quantile= True, ncols=3, figsize=(27,8), plot_params={'title': 'PDP of radius of gyration for WEEKDAYS',
                'subtitle': "3W1D",
                'title_fontsize': 15,
                'subtitle_fontsize': 12})
print(plt.show())

#Two Way Partial Dependence Plot (Latitude-Longitude)
pdp_combo = pdp.pdp_interact(
    model=model, dataset=X_test, model_features=X.columns, features=['avg_longitude', 'avg_latitude'], 
    num_grid_points=[10, 10],  percentile_ranges=[(6,91), (4,93)], n_jobs=1
)

fig, axes = pdp.pdp_interact_plot(
    pdp_combo, ['avg_longitude','avg_latitude'], plot_type='contour', x_quantile=True, ncols=2, 
    plot_pdp=True, which_classes=[2], figsize= (12,12), plot_params={'title': 'PDP interaction of Location Coordinates for WEEKDAYS', 'subtitle': '3W1D',
            'title_fontsize': 15}
            )

print(plt.show())
