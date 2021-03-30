import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib
from fitting import *

def mean_relative_error(y_true, y_pred):
    return sum(abs(y_pred-y_true)*100/y_true)/len(y_true)

def Train_SVR_all(train, X_columns, coefficient, kernel, epsilon, C, coef0=None):

    y_train = train["param_"+str(coefficient)].values
    
    
    scaler = MinMaxScaler().fit(train[X_columns])
    X_train = scaler.transform(train[X_columns])
    
    if coef0:
        model = SVR(kernel = kernel, epsilon = epsilon, C=C, coef0 = coef0)
    else: 
        model = SVR(kernel = kernel, epsilon = epsilon, C=C)
    model.fit(X_train, y_train)
    return model, scaler
    
def TestSVRCoef_all_drugs(test, X_columns, model, scaler, coefficient, training_details, 
                          print_results_as_string = True):
    X_test = scaler.transform(test[X_columns])
    test["pred_param_"+str(coefficient)] = model.predict(X_test) 
    
    y_test =  test["param_"+str(coefficient)].values
    
    if "DRUG_ID" not in test.columns:
        test.reset_index(inplace=True)
    drug_ids = list(test["DRUG_ID"].unique())
    
    #evaluate mae and mre for each drug profile
    test.set_index("DRUG_ID", inplace=True)
    
    mae = np.zeros(len(drug_ids))
    mre = np.zeros(len(drug_ids))
    for i, drug_id in list(enumerate(drug_ids)):
        
        try:
            y_test_drug = test.loc[drug_id, "param_"+str(coefficient)].values
            y_pred = test.loc[drug_id, "pred_param_"+str(coefficient)].values
            mae[i] = mean_absolute_error(y_test_drug, y_pred)
            mre[i] = mean_relative_error(y_test_drug, y_pred)
        except:
            y_test_drug = test.loc[drug_id, "param_"+str(coefficient)]
            y_pred = test.loc[drug_id, "pred_param_"+str(coefficient)]
            mae[i] = abs(y_test_drug - y_pred)
            mre[i] = (y_test_drug - y_pred)/y_test_drug
        
    mae_results = (mae.mean(), 2*mae.std())
    mre_results = (mre.mean(), 2*mre.std())
    if print_results_as_string:
        print("\nCoefficient %d, Modelling for %s\n"% (coefficient, training_details))
        print("MAE: %0.3f +/- %0.3f" % mae_results)
        print("MRE: %0.1f +/- %0.1f" % mre_results)
    else:
        return mae_results, mae_results
    

    
def ReconstructSVR_all_drugs(df, X_columns, fitting_function, conc_columns, response_columns,
                             n_coef = 4, recompute_predictions = False,
                             model_dict=None, scaler_dict=None, training_details=None, drop_columns=False):
    if "DRUG_ID" not in df.columns:
        df.reset_index(inplace=True)
    if recompute_predictions:
        pred_cols = []
        for i in range(1, n_coef+1):
            X_test = scaler_dict[i].transform(df[X_columns])
            col_name = "pred_param_"+str(i)
            df[col_name] = model_dict[i].predict(X_test)
            pred_cols.append(col_name)
    
    df = compute_r2_score(df, conc_columns, response_columns, 
                                             fitting_parameters = ["pred_param_"+str(i) for i in range(1, n_coef+1)], 
                                             fitting_function=fitting_function,
                                             return_fit_y=True)
    
    drop_cols= []
    dif_cols = []
    for i in range(len(response_columns)):
        new_col = "dif_"+str(i)
        df[new_col]= df[response_columns[i]] - df["pred_y_"+str(i)]
        dif_cols.append(new_col)
        drop_cols += ["pred_y_"+str(i), new_col]

    df["mae_pred"] = df[dif_cols].mean(axis=1)
    if drop_columns:   
        df.drop(drop_cols, axis=1, inplace=True)

    return df