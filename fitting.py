import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

import os, sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


def sigmoid_2_param(x, p, s):
    """ Sigmoid function from Dennis Wang's paper:
    x - dosage [0, 1],
    p - position,        default=0.4
    s - shape parameter, default=-1
    """
    return ( 1.0 / (1.0 + np.exp((x-p)/s)) )


def fsigmoid(x, p, k):
    """ Comparing with sigmoid_2_param:
    x = x  - dosage [0, 1]
    p - position [0,1],           default=0.4
    k = -1/s (s -shape parameter) default=0.4
    """
    return ( 1.0 / (1.0 + np.exp(-k*(x-p))) )

def sigmoid_3_param(x, x0, k, d):
    """ Comparing with sigmoid_2_param:
    x0 -  p - position, correlation with IC50 or EC50
    k = -1/s (s -shape parameter)
    d - determines the vertical position of the sigmoid - shift on y axis - better fitting then sigmoid_2_param
        """
    return ( 1/ (1 + np.exp(-k*(x-x0))) + d )


def sigmoid_4_param(x, x0, L, k, d):
    """ Comparing with sigmoid_2_param:
    x0 -  p - position, correlation with IC50 or EC50
    L = 1 in sigmoid_2_param, protect from devision by zero if x is too small 
    k = -1/s (s -shape parameter)
    d - determines the vertical position of the sigmoid - shift on y axis - better fitting then sigmoid_2_param

    """
    return ( 1/ (L + np.exp(-k*(x-x0))) + d)



def ll4_4_param(x, e, c, b, d):
    """ https://gist.github.com/yannabraham/5f210fed773785d8b638
    This function is basically a copy of the LL.4 function from the R drc package:
     - b: hill slope
     - d: min response - determines the vertical position of the graph
     - c: max response
     - e: EC50
     c-d - difference between max and min responses
     np.exp( b* (np.log(x)-np.log(e)) -  np.exp((x-p)/s in sigmoid_2_param
     b- hill slope = 1/s - shape parameter
     np.log(x)-np.log(e) == x-p in sigmoid_2_param
     """
    return ( (c-d)/(1 + np.exp( b*(np.log(x)-np.log(e) ))) + d)


def ll4R_4_param(x, c, a, b, d):
    """ LL.4 function from R
    https://www.rdocumentation.org/packages/drc/versions/2.5-12/topics/LL.4
   
    a-d - difference between max and min responses
    np.exp( b* np.log(x) - e) -  np.exp((x-p)/s in sigmoid_2_param
    b - hill slope = 1/s - shape parameter
    np.log(x)- e/b == x-p in sigmoid_2_param

    """
    return ( (a-d)/(1+np.exp(b*np.log(x)- c)) + d)


def logistic_4_param(x, A, B, C, d):
    """ https://people.duke.edu/~ccc14/pcfb/analysis.html
    4PL logistic equation
    sigmoid_2_param: 1.0 / (1.0 + np.exp((x-p)/s)
    (A - d) = 1 in sigmoid_2_param:
    (x/C)**B  - corresponds to np.exp((x-p)/s
    d - determines the vertical position of the graph
    """
    return ( (A-d)/(1.0+((x/C)**B)) + d )


def logLogist_3_param(x, EC50, HS, E_inf):
    """Python analog for PharmacoGx/R/LogLogisticRegression.R
    https://github.com/bhklab/PharmacoGx/blob/master/R/LogLogisticRegression.R
    E = E_inf + (1 - E_inf)/(1 + (x/EC50)^HS)
    sigmoid_2_param: 1.0 / (1.0 + np.exp((x-p)/s)
    
    (A - d) = 1 in sigmoid_2_param:
    (np.log10(x)/EC50)**HS  - (in logistic4 (x/C)**B) corresponds to np.exp((x-p)/s 
    
    E_inf - determines the vertical position of the graph /coefficient d, min response in other functions
    """
    return ((1-E_inf)/(1+(np.log10(x)/EC50)**HS) + E_inf)



def FitCurve(fitting_function, x, y, parameters_guess=None, to_plot = False):
#     from scipy.optimize import curve_fit

    if parameters_guess:
        parameters, p_covariance = curve_fit(fitting_function, x, y, parameters_guess)
    else: 
        parameters, p_covariance = curve_fit(fitting_function, x, y)
    x2 = np.linspace(0, 1, 10)
    y_fit = fitting_function(x, *parameters)
    r2 = r2_score(y, y_fit)

    if to_plot:
        print("Fitting parameters:", *parameters)
        plt.scatter(x, y)
        x2 = np.linspace(0, 1, 10)
        y2 = fitting_function(x2, *parameters)
        plt.plot(x2, y2, "blue", label = "R^2= %0.5f"%r2)   
        plt.title('Least-squares fit')
        plt.legend();
    return r2, parameters


def FittingColumn(df, indexes, x_columns, y_columns, fitting_function, parameters_guess=None, default_param = False):
    """
    intial parameter guess [max(y), np.median(x), 1, min(y)]
    potentially they can be different for each data row, but as soon as we have scaled and filtered data
    we can use by default [1.0, 0.4, 1.0, .0] 
    """
    
    r2_scores = np.zeros(len(indexes))
    X = df.loc[indexes, x_columns].values.astype(np.float32)
    Y = df.loc[indexes, y_columns].values.astype(np.float32)
    fitting_parameters = [None]*len(indexes)
    
    
    # parameters_guess= [np.median(x), 1, max(y), min(y)]
    default_param_model = {"sigmoid_2_param": [0.4, 0.1],
                       "fsigmoid" : [0.4, -10],
                       "sigmoid_4_param": [0.4, 1.0, 1.0, .0],
                       "sigmoid_3_param": [0.4, 1.0, .0],
                       "logistic_4_param": [1.0, 1.0, 1.0, 0.0],
                       "ll4_4_param": [0.4, 1.0, 1.0, 0.0],
                       "ll4R_4_param": [0.4, 1.0, 1.0, 0.0],
                       "logLogist_3_param": [-1, -0.1, 0.1]}
    
    if default_param:
        parameters_guess = default_param_model[fitting_function]
       
    else:
        pass
    
    for i in tqdm(range(len(indexes))):
        x = X[i, :]
        y = Y[i, :]
    
        try:
            r2_scores[i], fitting_parameters[i] = FitCurve(fitting_function_object, x, y, parameters_guess = parameters_guess)
            
        except:
            try:
                functions = {"fsigmoid": fsigmoid, 
                 "sigmoid_2_param": sigmoid_2_param, 
                "sigmoid_4_param": sigmoid_4_param,
                 "sigmoid_3_param": sigmoid_3_param, 
                 "logistic_4_param": logistic_4_param,  
                 "ll4_4_param": ll4_4_param, 
                 "ll4R_4_param":ll4R_4_param,
                 "logLogist_3_param":logLogist_3_param}
                fitting_function_object = functions[fitting_function]
                r2_scores[i], fitting_parameters[i] = FitCurve(fitting_function_object, x, y, 
                                                               parameters_guess = parameters_guess)
            except:
                r2_scores[i] = 0
    print(fitting_function_object)
    return r2_scores, fitting_parameters


def ShowResponseCurvesWithFitting(df, plots_in_row, plots_in_column, x_columns, y_columns, start_index=0, indexes=[],
                         fitting_function =None, fitting_parameters = None, pred_fitting_param = None, drug_dict = None,
                                  CCL_dict=None):
    
    print_general_title = False
    fig = plt.figure(figsize=(14, 11))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    n_plots= plots_in_row*plots_in_column
    
    if len(indexes) == 0:
        indexes = df.index[start_index : start_index+n_plots]
        
    for i, ind in list(enumerate(indexes)):
        
        x = df.loc[ind, x_columns].values.astype(np.float32)
        y = df.loc[ind, y_columns].values.astype(np.float32)

                
        ax = fig.add_subplot(plots_in_row, plots_in_column, i+1)
        ax.scatter(x,y)
        if drug_dict and CCL_dict:
            ax.set_title("Drug: "+ drug_dict[df.loc[ind, "DRUG_ID"]]+" / CCL: "+CCL_dict[df.loc[ind, "COSMIC_ID"]])
        elif drug_dict:
            ax.set_title("Drug: "+drug_dict[df.loc[ind, "DRUG_ID"]] +"_"+ str(df.loc[ind, "COSMIC_ID"]))
        elif ("drug_name" in df.columns) and ("CCL_name" in df.columns):
            ax.set_title("Drug: "+ str(df.loc[ind, "drug_name"])+" / CCL: "+ str(df.loc[ind, "CCL_name"]))
        else:
            print_general_title = True
            ax.set_title(str(df.loc[ind, "DRUG_ID"])+"_"+str(df.loc[ind, "COSMIC_ID"]))
                    
        ax.set_xlabel("Scaled dosage")
        ax.set_ylabel("Normalised response")
        
        if fitting_function:
            functions = {"fsigmoid": fsigmoid, 
                             "sigmoid_2_param": sigmoid_2_param, 
                            "sigmoid_4_param": sigmoid_4_param,
                             "sigmoid_3_param": sigmoid_3_param, 
                             "logistic_4_param": logistic_4_param,  
                             "ll4_4_param": ll4_4_param, 
                             "ll4R_4_param":ll4R_4_param,
                             "logLogist_3_param":logLogist_3_param}
                
            fitting_function_object = functions[fitting_function]
            
            x2 = np.linspace(0, 1, 10)

            if type(fitting_parameters) == str:
                fit_param = df.loc[ind, fitting_parameters]
                if len(fit_param) > 5:
                    fit_param = [float(i) for i in fit_param.strip("[ ").strip(" ]").split()]
            else:
                fit_param = df.loc[ind, fitting_parameters].values


            y_fit = fitting_function_object(x, *fit_param)
            y2 = fitting_function_object(x2, *fit_param)
            
            r2 = r2_score(y, y_fit)
            ax.plot(x2, y2, label= "R^2 fit = %0.4f"% r2)
            ax.legend()
                
        if pred_fitting_param:
            x3 = np.linspace(0, 1, 10) 
            fit_param = df.loc[ind, pred_fitting_param]    
            y_fit3 = fitting_function_object(x, *fit_param)
            y3 = fitting_function_object(x3, *fit_param)
            r2_pred = r2_score(y, y_fit3)
            ax.plot(x3, y3, color="red", label= "R^2 pred = %0.4f"% r2_pred)
            ax.legend()

        
    if print_general_title:
        print("Figures titles: Index_DRUG_ID_COSMIC_ID (COSMIC_ID is a cell line)")
        
def compute_r2_score(df, x_columns, y_columns, fitting_parameters, fitting_function="sigmoid_4_param"):
    functions = {"fsigmoid": fsigmoid, 
                 "sigmoid_2_param": sigmoid_2_param, 
                "sigmoid_4_param": sigmoid_4_param,
                 "sigmoid_3_param": sigmoid_3_param, 
                 "logistic_4_param": logistic_4_param,  
                 "ll4_4_param": ll4_4_param, 
                 "ll4R_4_param":ll4R_4_param,
                 "logLogist_3_param":logLogist_3_param}
    
    fitting_function_object = functions[fitting_function]
    r2_scores=np.zeros(len(df.index))
    for i in range(len(df.index)):
        x = df.loc[df.index[i], x_columns]
        y = df.loc[df.index[i], y_columns]
        if type(fitting_parameters) == str:
            fit_param = df.loc[df.index[i], fitting_parameters]
        else:
            fit_param = df.loc[df.index[i], fitting_parameters].values
        y_fit = fitting_function_object(x, *fit_param)
        r2_scores[i] = r2_score(y, y_fit)
    return r2_scores

def ShowOneFitting(df, ind, conc_columns, response_norm, fitting_function, fitting_parameters,
                   predicted_param=None, save_fig_name=None, fig_size=None):
    
    if fig_size:
        plt.figure(figsize=fig_size)
    else:
        plt.figure(figsize=(4,4))
    x = df.loc[ind, conc_columns].astype("float32")
    y = df.loc[ind, response_norm].astype("float32")
    plt.scatter(x, y)
    plt.tick_params(labelsize=14)
    plt.xlabel("Scaled dosage", fontsize=14)
    plt.ylabel("Normalised Response", fontsize=14)
    fit_param = df.loc[ind, fitting_parameters]

    y_fit = sigmoid_4_param(x, *fit_param)
    r2 = r2_score(y, y_fit)
    plt.plot(x, y_fit, label= "fitting: R2= "+str(round(r2, 3)))
    if predicted_param:
        print("Fitting parameters:", fit_param)
        print("Predicted parameters:", predicted_param)
        y_fit = sigmoid_4_param(x, *predicted_param)
        r2 = r2_score(y, y_fit)
        plt.plot(x, y_fit, label= "predicted : R2= "+str(round(r2, 3)))

    plt.legend()
    if save_fig_name:
        plt.savefig(save_fig_name, bbox_inches='tight', dpi=300);
        
def ComputeFittingFunction(df, fitting_function, x_columns, y_columns):
    shape_1 = df.shape[0]
    r2, fit_param = FittingColumn(df, df.index, x_columns,y_columns,
                               fitting_function = fitting_function, default_param=True)
    df[fitting_function+"_r2"] = r2
    df[fitting_function] = fit_param
    df = df[df[fitting_function].isnull()==False]
    if df.shape[0]!= shape_1:
        print("Reduced number of samples:", shape_1 - df.shape[0])
    return df
    
        
def CompareFittingFunctions(df, functions, conc_columns, response_norm, recompute_fitting= True, save_file_name=None):
    print(df.shape)
    if recompute_fitting:
        for fitting_function in functions:
            print("\n", fitting_function)
            ComputeFittingFunction(df, fitting_function, conc_columns, response_norm)
        

    functions_dict= dict(list(enumerate(functions)))
    r2_columns = [fitting_function+"_r2" for fitting_function in functions]

    df["better_fitting"] = np.argmax(df[r2_columns].values, axis=1)
    r2_col_res = r2_columns +["better_fitting"]
    df["better_fitting"] = df["better_fitting"].map(functions_dict)
    # df[r2_col_res].head()

    print("\n")
    best_functions = df["better_fitting"].unique()

    df_best = pd.DataFrame(index= functions)
    for fitting_function in functions:
        r2_fit = df[fitting_function+"_r2"].values 
        try:
            df_best.loc[fitting_function, "best_fitting_count"] = df[df["better_fitting"]==fitting_function].shape[0]
        except:
             df_best.loc[fitting_function, "best_fitting_count"] = 0
        df_best.loc[fitting_function, "min"] = min(r2_fit)
        df_best.loc[fitting_function, "max"] = max(r2_fit)
        df_best.loc[fitting_function, "r2>0"] = (r2_fit >0).sum().astype("int32")
        df_best.loc[fitting_function, "r2>0.8"] = (r2_fit >0.8).sum().astype("int32")
        df_best.loc[fitting_function, "r2>0.9"] = (r2_fit >0.9).sum().astype("int32")
        df_best.loc[fitting_function, "r2>0.99"] = (r2_fit >0.9).sum().astype("int32")
    display(df_best)
    print("\nExamples of bad fitting with sigmoid_4_param (r2<0.61):", df[df["sigmoid_4_param_r2"]<0.61].shape[0])
    display(df[df["sigmoid_4_param_r2"]<0.61][["COSMIC_ID", "DRUG_ID"]+r2_col_res].head())
    if save_file_name:
        df.to_csv(save_file_name, index=False)
    return df
        