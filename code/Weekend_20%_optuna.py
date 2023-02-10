import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score
import optuna
import xgboost as xgb
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import shap
import matplotlib.pyplot as plt
from pdpbox.pdp import pdp_isolate, pdp_interact, pdp_interact_plot, pdp_plot
from pdpbox import pdp, get_dataset, info_plots

# function of optuna
def objective(trial):
    check_acc=0
    param = {
        "verbosity": 0,
        "objective": 'multi:softmax',
        'num_class':3,
        #'multi:softprob'
        # use exact for small dataset.
        "tree_method": "exact",
        # defines booster, gblinear for linear functions.
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        # L2 regularization weight.
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }

    if param["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)


    x_general_gk = X_train_general.to_numpy()
    y_general_gk = y_train_general.to_numpy()
   
    #Set to ten groups--------------------------------------
    kfold = KFold(n_splits=10,shuffle=True)
    sm = SMOTE(random_state = 3,sampling_strategy='not majority')
    for train_index, test_index in kfold.split(x_general_gk):
        #Cross-tabulation of the train dataset
        sub_x_train_general, sub_x_test_general = x_general_gk[train_index], x_general_gk[test_index]
        sub_y_train_general, sub_y_test_general = y_general_gk[train_index], y_general_gk[test_index]
        try:
            sub_x_train_general, sub_y_train_general = sm.fit_resample(sub_x_train_general, sub_y_train_general)
        except:
            print('')
        
        dtrain = xgb.DMatrix(sub_x_train_general, label=sub_y_train_general)
        dvalid = xgb.DMatrix(sub_x_test_general, label=sub_y_test_general)

        bst = xgb.train(param, dtrain)
        preds = bst.predict(dvalid)
        pred_labels = np.rint(preds)
        f1=f1_score(sub_y_test_general, pred_labels, average='micro')
        check_acc=check_acc+f1

    return (check_acc/10)

#function of analysis
def xgboostML(params,lines,X_train_general, X_test_general, y_train_general, y_test_general):    
    model = XGBClassifier(**params)
    try:
        sm = SMOTE(random_state = 3,sampling_strategy='not majority')
        X_train_general, y_train_general = sm.fit_resample(X_train_general, y_train_general) 
    except:
        print('')
    features= X_test_general.columns
    X_train_general.columns=features
    model.fit(X_train_general,y_train_general)


    y_pred = model.predict(X_test_general)
    score=accuracy_score(y_pred,y_test_general)
    list_res_temp.append(score)
    list_res_temp.append(cohen_kappa_score(y_test_general, y_pred))
    list_res_temp.append(f1_score(y_test_general, y_pred, average='micro'))


    res_pre_f1=classification_report(y_test_general, y_pred)
    print(res_pre_f1)
    res_pre_f1_1=res_pre_f1.split('         ')
    res_pre_f1_N=res_pre_f1_1[2].split('      ')
    res_pre_f1_D=res_pre_f1_1[3].split('      ')
    res_pre_f1_BD=res_pre_f1_1[4].split('      ')

    list_res_temp.append(float(res_pre_f1_N[1]))
    list_res_temp.append(float(res_pre_f1_N[2]))
    list_res_temp.append(float(res_pre_f1_N[3]))

    try:
        list_res_temp.append(float(res_pre_f1_D[1]))
        list_res_temp.append(float(res_pre_f1_D[2]))
        list_res_temp.append(float(res_pre_f1_D[3]))
    except:
        res_pre_f1_D=res_pre_f1_1[4].split('      ')
        if res_pre_f1_D[0]==2:
            list_res_temp.append(float(0))
            list_res_temp.append(float(0))
            list_res_temp.append(float(0))
            res_pre_f1_BD=res_pre_f1_1[4].split('      ')
        else:
            res_pre_f1_BD=res_pre_f1_1[5].split('      ')
            list_res_temp.append(float(res_pre_f1_D[1]))
            list_res_temp.append(float(res_pre_f1_D[2]))
            list_res_temp.append(float(res_pre_f1_D[3]))


    try:
        list_res_temp.append(float(res_pre_f1_BD[1]))
        list_res_temp.append(float(res_pre_f1_BD[2]))
        list_res_temp.append(float(res_pre_f1_BD[3]))
    except:
        try:
            res_pre_f1_BD=res_pre_f1_1[5].split('      ')
            list_res_temp.append(float(res_pre_f1_BD[1]))
            list_res_temp.append(float(res_pre_f1_BD[2]))
            list_res_temp.append(float(res_pre_f1_BD[3]))
        except:
            res_pre_f1_BD=res_pre_f1_1[6].split('      ')
            list_res_temp.append(float(res_pre_f1_BD[1]))
            list_res_temp.append(float(res_pre_f1_BD[2]))
            list_res_temp.append(float(res_pre_f1_BD[3]))


    list_res_temp.append(lines)
    list_res_temp.append(params)
    """
    You can select a smaller subset of X_test_general to analyze for quick results. 
    """
    explainer = shap.KernelExplainer(model.predict_proba, X_test_general)
    shap_values = explainer.shap_values(X_test_general)

    #Beeswarm Summary Plot
    print(shap.summary_plot(shap_values[2], X_test_general, max_display=20))
    #Feature Importance Bar Plot
    print(shap.summary_plot(shap_values[2], X_test_general, max_display=20, plot_type='bar'))

    #SHAP Dependence Plot 
    print(shap_figure=shap.dependence_plot('radius_of_gyration', shap_values[2], X_test_general, interaction_index='time_of_day'))

    #Partial Dependence Plot (Radius of Gyration)
    pdp_dist = pdp.pdp_isolate(model=model, dataset=X_test_general, model_features=X_test_general.columns, feature='radius_of_gyration', num_grid_points=11)
    pdp.pdp_plot(pdp_dist, 'radius_of_gyration', x_quantile= True, ncols=3, figsize=(27,8), plot_params={'title': 'PDP of radius of gyration for WEEKDAYS',
                    'subtitle': "3W1D",
                    'title_fontsize': 15,
                    'subtitle_fontsize': 12})
    print(plt.show())

    #Two Way Partial Dependence Plot (Latitude-Longitude)
    pdp_combo = pdp.pdp_interact(
        model=model, dataset=X_test_general, model_features=X_test_general.columns, features=['avg_longitude', 'avg_latitude'], 
        num_grid_points=[10, 10],  percentile_ranges=[(6,91), (4,93)], n_jobs=1
    )

    fig, axes = pdp.pdp_interact_plot(
        pdp_combo, ['avg_longitude','avg_latitude'], plot_type='contour', x_quantile=True, ncols=2, 
        plot_pdp=True, which_classes=[2], figsize= (12,12), plot_params={'title': 'PDP interaction of Location Coordinates for WEEKDAYS', 'subtitle': '3W1D',
                'title_fontsize': 15}
                )

    print(plt.show())

    df_res.loc[0]=list_res_temp
    df_res.to_csv('%s_weekend_result.csv'%str_name[0])


def main():
    global str_name
    global X_train_general
    global X_test_general
    global y_train_general
    global y_test_general 
    global list_res_temp

    # Data Loading
    print('Data Loading ...')
    file_name = '15_result_indexed_dataset_window3_distance1.csv'
    df=pd.read_csv(file_name)
    str_name = file_name.split('.')
    df['WTSD_latitude'].fillna(0, inplace=True)
    df['WTSD_longitude'].fillna(0, inplace=True)
    df_omit=df.dropna()
    df_omit = df_omit[df_omit['day_of_week'] >4]
    df_omit=df_omit.drop(columns='day_of_week')
    df_omit["label"] = pd.factorize(df_omit["label"])[0].astype(int)
    df_omit=df_omit.drop(columns='pid')
    df_omit=df_omit.drop(columns='study_duration')
    df_omit=df_omit.drop(columns='time_stamp') 
    y_general = df_omit['label'] 
    x_general_1 = df_omit.drop(columns='label')
    list_res_temp.append(str_name[0])
    

    # Maxmin the data
    print('Maxmin the data ...')
    list_gereral=(list(x_general_1))
    scaler = MinMaxScaler()
    scaler.fit(x_general_1)
    x_general_1 = scaler.transform(x_general_1)
    x_general_1=pd.DataFrame(x_general_1)
    x_general_1.columns=list_gereral
    

    # Data split
    print('Data split ...')
    X_train_general, X_test_general, y_train_general, y_test_general= train_test_split(x_general_1, y_general,test_size=0.2)
    

    #Use optuna to train the best param
    print('Use optuna to train the best param ...')
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=600)
    trial = study.best_trial
    params=trial.params
    lines=len(df_omit)

    #Data Analysis
    print('Data Analysis ...')
    xgboostML(params,lines,X_train_general, X_test_general, y_train_general, y_test_general)



# Generate null dataset to save result
df_res = pd.DataFrame(columns=['name','score','Kappa','F1-score','precision_N',"recall_N",'f1-score_N','precision_D',"recall_D",'f1-score_D','precision_BD',"recall_BD",'f1-score_BD','line','params'])
list_res_temp=[]

if __name__ == '__main__':
    main()
    print('Start ...')