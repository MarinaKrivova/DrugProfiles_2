import pandas as pd
import numpy as np
import os

import seaborn as sns
import matplotlib.pyplot as plt
import gc
import time

import warnings
warnings.filterwarnings("ignore")

from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from all_functions import DataPreprocessing, TrainTestSplit

np.random.seed(123)

_FOLDER = "results/"



def mean_relative_error(y_true, y_pred):
    return sum(abs(y_pred-y_true)*100/y_true)/len(y_true)

def TrainTest_KR_Alg1(train, test, target, drug_ids_list, X_columns, kernel, hyperparameters, 
                    degree = 3, print_drug_name = True):
    
    mae = np.zeros(len(drug_ids_list))
    mre = np.zeros(len(drug_ids_list))
    y_pred_all = []
    for i, drug_id in list(enumerate(drug_ids_list)):
        if print_drug_name:
            drug_name = train.loc[drug_id, "Drug_Name"].values[0]
            print(drug_id, drug_name)
        train_drug = train.loc[drug_id,:]
        test_drug = test.loc[drug_id,:]
        y_train_drug = train_drug[target].values
        y_test_drug =  test_drug[target].values
        scaler = MinMaxScaler().fit(train_drug[X_columns])
        Xtrain_drug = scaler.transform(train_drug[X_columns])
        if "best_degree3" in hyperparameters.columns:
            model = KernelRidge(kernel = kernel, 
                        alpha = hyperparameters.loc[drug_id, "best_alpha3"], 
                        gamma = hyperparameters.loc[drug_id, "best_gamma3"],
                        coef0= hyperparameters.loc[drug_id, "best_coef03"], 
                        degree = hyperparameters.loc[drug_id, "best_degree3"])
        else: 
            model = KernelRidge(kernel = kernel, 
                        alpha = hyperparameters.loc[drug_id, "best_alpha3"], 
                        gamma = hyperparameters.loc[drug_id, "best_gamma3"],
                        coef0= hyperparameters.loc[drug_id, "best_coef03"])
            
        model.fit(Xtrain_drug, y_train_drug)
        Xtest_drug = scaler.transform(test_drug[X_columns])
        y_pred = (model.predict(Xtest_drug))
        mae[i] = mean_absolute_error(y_test_drug, y_pred)
        mre[i] = mean_relative_error(y_test_drug, y_pred)
        y_pred_all.extend(y_pred)
        
    mean_mae = str(round(mae.mean(), 3))+ " +/- " + str(round(mae.std(), 3))
    mean_mre = str(round(mre.mean(), 1))+ " +/- " + str(round(mre.std(), 1))
    print("\nMAE:", mean_mae)
    print("MRE:", mean_mre)
    print("")
    print(train.shape, test.shape)
    return mean_mae, mean_mre, y_pred_all, train.shape, test.shape

def TrainTest_SVR_Alg1(train, test, target, drug_ids_list, X_columns, kernel, hyperparameters, 
                    degree = 3, print_drug_name = True):
    
    mae = np.zeros(len(drug_ids_list))
    mre = np.zeros(len(drug_ids_list))
    y_pred_all = []
    for i, drug_id in list(enumerate(drug_ids_list)):
        if print_drug_name:
            drug_name = train.loc[drug_id, "Drug_Name"].values[0]
            print(drug_id, drug_name)
        train_drug = train.loc[drug_id,:]
        test_drug = test.loc[drug_id,:]
        y_train_drug = train_drug[target].values
        y_test_drug =  test_drug[target].values
        scaler = MinMaxScaler().fit(train_drug[X_columns])
        Xtrain_drug = scaler.transform(train_drug[X_columns])
        
        if "best_coef03" in hyperparameters.columns:
            model = SVR(kernel = kernel, 
                C= hyperparameters.loc[drug_id, "best_C3"], 
                epsilon = hyperparameters.loc[drug_id, "best_epsilon3"],
                coef0= hyperparameters.loc[drug_id, "best_coef03"])
        else:
            model = SVR(kernel = kernel, 
                C= hyperparameters.loc[drug_id, "best_C3"], 
                epsilon = hyperparameters.loc[drug_id, "best_epsilon3"])
        
        model.fit(Xtrain_drug, y_train_drug)
        Xtest_drug = scaler.transform(test_drug[X_columns])
        y_pred = (model.predict(Xtest_drug))
        mae[i] = mean_absolute_error(y_test_drug, y_pred)
        mre[i] = mean_relative_error(y_test_drug, y_pred)
        y_pred_all.extend(y_pred)
        
    mean_mae = str(round(mae.mean(), 3))+ " +/- " + str(round(mae.std(), 3))
    mean_mre = str(round(mre.mean(), 1))+ " +/- " + str(round(mre.std(), 1))
    print("\nMAE:", mean_mae)
    print("MRE:", mean_mre)
    print("")
    print(train.shape, test.shape)
    return mean_mae, mean_mre, y_pred_all, train.shape, test.shape

def TrainTest_SVR_subset_Alg1(train, test, target, drug_ids_list, subset_feat_dict, kernel, hyperparameters, 
                    degree = 3, print_drug_name = True):
    
    mae = np.zeros(len(drug_ids_list))
    mre = np.zeros(len(drug_ids_list))
    y_pred_all = []
    for i, drug_id in list(enumerate(drug_ids_list)):
        if print_drug_name:
            drug_name = train.loc[drug_id, "Drug_Name"].values[0]
            print(drug_id, drug_name)
        train_drug = train.loc[drug_id,:]
        test_drug = test.loc[drug_id,:]
        y_train_drug = train_drug[target].values
        y_test_drug =  test_drug[target].values
        X_columns = subset_feat_dict[drug_id]
        scaler = MinMaxScaler().fit(train_drug[X_columns])
        Xtrain_drug = scaler.transform(train_drug[X_columns])
        if "best_coef03" in hyperparameters.columns:
            model = SVR(kernel = kernel, 
                C= hyperparameters.loc[drug_id, "best_C3"], 
                epsilon = hyperparameters.loc[drug_id, "best_epsilon3"],
                coef0= hyperparameters.loc[drug_id, "best_coef03"])
        else:
            model = SVR(kernel = kernel, 
                C= hyperparameters.loc[drug_id, "best_C3"], 
                epsilon = hyperparameters.loc[drug_id, "best_epsilon3"])
        
        model.fit(Xtrain_drug, y_train_drug)
        Xtest_drug = scaler.transform(test_drug[X_columns])
        y_pred = (model.predict(Xtest_drug))
        mae[i] = mean_absolute_error(y_test_drug, y_pred)
        mre[i] = mean_relative_error(y_test_drug, y_pred)
        y_pred_all.extend(y_pred)
        
    print("\nMAE:", round(mae.mean(), 3), "+/-", round(mae.std(), 3))
    print("MRE:", round(mre.mean(), 1), "+/-", round(mre.std(), 1))
    print("")
    print(train.shape, test.shape)
    return mae, mre, y_pred_all


def TrainTest_KR_subset_Alg_1(train, test, target, drug_ids_list, subset_feat_dict, kernel, hyperparameters, 
                    degree = 3, print_drug_name = True):
    mae = np.zeros(len(drug_ids_list))
    mre = np.zeros(len(drug_ids_list))
    y_pred_all = []
    for i, drug_id in list(enumerate(drug_ids_list)):
        if print_drug_name:
            drug_name = train.loc[drug_id, "Drug_Name"].values[0]
            print(drug_id, drug_name)
        train_drug = train.loc[drug_id,:]
        test_drug = test.loc[drug_id,:]
        y_train_drug = train_drug[target].values
        y_test_drug =  test_drug[target].values
        X_columns = subset_feat_dict[drug_id]
        scaler = MinMaxScaler().fit(train_drug[X_columns])
        Xtrain_drug = scaler.transform(train_drug[X_columns])
        
        model = KernelRidge(kernel = kernel, 
                        alpha = hyperparameters.loc[drug_id, "best_alpha3"], 
                        gamma = hyperparameters.loc[drug_id, "best_gamma3"],
                        coef0= hyperparameters.loc[drug_id, "best_coef03"], 
                        degree = hyperparameters.loc[drug_id, "best_degree3"])
        model.fit(Xtrain_drug, y_train_drug)        
        Xtest_drug = scaler.transform(test_drug[X_columns])
        y_pred = (model.predict(Xtest_drug))
        mae[i] = mean_absolute_error(y_test_drug, y_pred)
        mre[i] = mean_relative_error(y_test_drug, y_pred)
        y_pred_all.extend(y_pred)
        
    print("\nMAE:", round(mae.mean(), 3), "+/-", round(mae.std(), 3))
    print("MRE:", round(mre.mean(), 1), "+/-", round(mre.std(), 1))
    print("")
    print(train.shape, test.shape)
    return mae, mre, y_pred_all

import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

def ShowErrors(target_coef, y_pred, test_df, ml_scenario, save_fig = False, folder_to_save= None, name_to_save = "coef_1.png"):
    test_df["y_pred"] = y_pred
    test_df["abs_error"]= abs(test_df[target_coef] - y_pred) 
    
    csfont = {'fontname':'Times New Roman'}
    plt.figure(figsize=(4, 3))
    sns.boxplot(x = "Drug_Name", y = "abs_error", data = test_df)

    plt.xticks(rotation = 90, fontsize=12, **csfont)
    plt.yticks(fontsize=12, **csfont)
    plt.ylabel("Absolute Error", fontsize=12, **csfont)
    plt.title(ml_scenario)
    if save_fig:
        plt.savefig(folder_to_save + name_to_save, bbox_inches='tight');
        
def CompareTrainingScenarios(target_coef, tested_scenarios, kernel, hyperparameters, X_columns, 
                             folder_with_original_data, folder_with_results,
                             model_type= "KR_alg1", print_progress_info = False):
    
    df_results = pd.DataFrame(columns=['ml_scenario', 'mae', 'mre', "train_test_shape"]).set_index("ml_scenario")
   
    for ml_scenario in tested_scenarios:
        df= DataPreprocessing(folder_with_original_data= folder_with_original_data, 
                              folder_with_results =folder_with_results, 
                              filtering_scenario = tested_scenarios[ml_scenario]['filtering'],
                              first_points_lower_limit = 0.8, last_points_upper_limit = 0.4,
                              middle_points_limit =-0.1, keep_r2_column = True,
                              fitting_function = tested_scenarios[ml_scenario]["fitting_function"], 
                              print_progress_info = print_progress_info)

        drug_ids_limit, train_df_limit, test_df_limit = TrainTestSplit(df, min_number_drug_profiles =50, 
                            train_ratio= 0.8, r2_restriction = tested_scenarios[ml_scenario]["r2_restriction"], 
                            print_progress_info = print_progress_info)
        print(ml_scenario)
        
        if model_type == "KR_alg1":
            mae, mre, y_pred, train_shape, test_shape = TrainTest_KR_Alg1(train_df_limit, test_df_limit, 
                        target = "param_"+str(target_coef), 
                      drug_ids_list =drug_ids_limit, X_columns = X_columns, 
                      kernel = kernel, hyperparameters= hyperparameters, print_drug_name = False)
        elif model_type == "SVR_alg1":
            mae, mre, y_pred, train_shape, test_shape = TrainTest_SVR_Alg1(train_df_limit, test_df_limit, 
                        target = "param_"+str(target_coef), 
                      drug_ids_list =drug_ids_limit, X_columns = X_columns, 
                      kernel = kernel, hyperparameters= hyperparameters, print_drug_name = False)
        else:
            print("ERROR: Unknown model")

        df_results.loc[ml_scenario] = [mae, mre, str(train_shape[0])+" , " +str(test_shape[0])]

        ShowErrors("param_"+str(target_coef), y_pred, test_df_limit, ml_scenario)
        del df
    return df_results



def CompareTrainingScenariosModels(target_coef, models_dict, X_columns, 
                             folder_with_original_data, folder_with_results,
                             fitting_functions, filtering_scenarios, r2_restrictions,
                             print_progress_info = False):
    
    tested_scenarios = {}

    for model in models_dict:
        for filtering in filtering_scenarios:
            for r in r2_restrictions:
                scenario_name = "Filtering " + str(filtering)+", r2>"+ str(r)+ ", "+model
                tested_scenarios[scenario_name]={}
                tested_scenarios[scenario_name]["filtering"] = filtering
                tested_scenarios[scenario_name]["r2_restriction"] = r
                tested_scenarios[scenario_name]["model_type"] = model

    dict_results = {}
    scenario_short = ["Filtering " + str(filtering)+", r2>"+ str(r)+ ", "+model for filtering in filtering_scenarios
                      for r in r2_restrictions for model in models_dict]

    for fitting_function in fitting_functions:
        df_results = pd.DataFrame(index= np.array(scenario_short))
        for model in models_dict:
            for filtering in filtering_scenarios:
                for r in r2_restrictions:
                    scenario_name = "Filtering " + str(filtering)+", r2>"+ str(r)+ ", "+model
                    df= DataPreprocessing(folder_with_original_data= folder_with_original_data, 
                                  folder_with_results =folder_with_results, 
                                  filtering_scenario = filtering,
                                  first_points_lower_limit = 0.8, last_points_upper_limit = 0.4,
                                  middle_points_limit =-0.1, keep_r2_column = True,
                                  fitting_function = fitting_function, 
                                  print_progress_info = print_progress_info)
#                     print("param_"+str(target_coef) in df.columns)

                    drug_ids_limit, train_df_limit, test_df_limit = TrainTestSplit(df, min_number_drug_profiles =50, 
                                                            train_ratio= 0.8, r2_restriction = r, 
                                                            print_progress_info = print_progress_info)
                    print(fitting_function, scenario_name)
                
                    hyperparameters = models_dict[model]["hyperparameters"]
                    kernel = models_dict[model]["kernel"]
                    model_type = models_dict[model]['model_type']
        
                    if model_type == "KR_alg1":
                        mae, mre, y_pred, train_shape, test_shape = TrainTest_KR_Alg1(train_df_limit, test_df_limit, 
                                                                target = "param_"+str(target_coef), 
                                                                drug_ids_list =drug_ids_limit, X_columns = X_columns, 
                                                                kernel = kernel, hyperparameters= hyperparameters, 
                                                                print_drug_name = False)
                    elif model_type == "SVR_alg1":
                        mae, mre, y_pred, train_shape, test_shape = TrainTest_SVR_Alg1(train_df_limit, test_df_limit, 
                                                                target = "param_"+str(target_coef), 
                                                                drug_ids_list =drug_ids_limit, X_columns = X_columns, 
                                                                kernel = kernel, hyperparameters= hyperparameters, 
                                                                print_drug_name = False)
                    else:
                        print("ERROR: Unknown model")

                    df_results.loc[scenario_name, "mae_coef_"+str(target_coef)] = mae
                    df_results.loc[scenario_name, "mre_coef_"+str(target_coef)] = mre
                    df_results.loc[scenario_name, "train_test_shape"] = str(train_shape[0])+" , " +str(test_shape[0])

        print("\n", fitting_function)
        dict_results[fitting_function]= df_results
        display(df_results)
        ShowErrors("param_"+str(target_coef), y_pred, test_df_limit, scenario_name)
        del df

    return dict_results