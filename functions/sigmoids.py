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


def content_sigmoid_functions():
    """
    All the functions in the current file
    """
    all_functions = {
        "sigmoid_2_param": "returns y = 1.0 / (1.0 + np.exp((x-p)/s)",
        "fsigmoid": "1.0 / returns y = (1.0 + np.exp(-k*(x-p)))",
        "sigmoid_3_param": "returns y = 1/ (1 + np.exp(-k*(x-x0))) + d",
        "sigmoid_4_param": "returns y =1/ (L + np.exp(-k*(x-x0))) + d",
        "ll4_4_param": " returns y = (c-d)/(1 + np.exp( b*(np.log(x)-np.log(e) ))) + d",
        "ll4R_4_param": "returns y = (a-d)/(1+np.exp(b*np.log(x)- c)) + d",
        "logistic_4_param": "returns y = (A-d)/(1.0+((x/C)**B)) + d ",
        "logLogist_3_param": "returns y = (1-E_inf)/(1+(np.log10(x)/EC50)**HS) + E_inf",
    }
    return all_functions


def sigmoid_2_param(x, p, s):
    """
    Returns y = 1.0 / (1.0 + np.exp((x-p)/s)
    Sigmoid function from Dennis Wang's paper:
    x - dosage [0, 1],
    p - position,        default=0.4
    s - shape parameter, default=-1
    """
    return 1.0 / (1.0 + np.exp((x - p) / s))


def fsigmoid(x, p, k):
    """Returns y = (1.0 + np.exp(-k*(x-p)))
    Comparing with sigmoid_2_param:
    x = x  - dosage [0, 1]
    p - position [0,1],           default=0.4
    k = -1/s (s -shape parameter) default=0.4
    """
    return 1.0 / (1.0 + np.exp(-k * (x - p)))


def sigmoid_3_param(x, x0, k, d):
    """
    Returns y = 1/ (1 + np.exp(-k*(x-x0))) + d
    Comparing with sigmoid_2_param:
    x0 -  p - position, correlation with IC50 or EC50
    k = -1/s (s -shape parameter)
    d - determines the vertical position of the sigmoid - shift on y axis - better fitting then sigmoid_2_param
    """
    return 1 / (1 + np.exp(-k * (x - x0))) + d


def sigmoid_4_param(x, x0, L, k, d):
    """Returns y =1/ (L + np.exp(-k*(x-x0))) + d
    Comparing with sigmoid_2_param:
    x0 -  p - position, correlation with IC50 or EC50
    L = 1 in sigmoid_2_param, protect from devision by zero if x is too small
    k = -1/s (s -shape parameter)
    d - determines the vertical position of the sigmoid - shift on y axis - better fitting then sigmoid_2_param

    """
    return 1 / (L + np.exp(-k * (x - x0))) + d


def ll4_4_param(x, e, c, b, d):
    """
    Returns y = (c-d)/(1 + np.exp( b*(np.log(x)-np.log(e) ))) + d
    https://gist.github.com/yannabraham/5f210fed773785d8b638
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
    return (c - d) / (1 + np.exp(b * (np.log(x) - np.log(e)))) + d


def ll4R_4_param(x, c, a, b, d):
    """
    Returns y = (a-d)/(1+np.exp(b*np.log(x)- c)) + d
    LL.4 function from R
    https://www.rdocumentation.org/packages/drc/versions/2.5-12/topics/LL.4

    a-d - difference between max and min responses
    np.exp( b* np.log(x) - e) -  np.exp((x-p)/s in sigmoid_2_param
    b - hill slope = 1/s - shape parameter
    np.log(x)- e/b == x-p in sigmoid_2_param
    """
    return (a - d) / (1 + np.exp(b * np.log(x) - c)) + d


def logistic_4_param(x, A, B, C, d):
    """
    Returns y = (A-d)/(1.0+((x/C)**B)) + d
    https://people.duke.edu/~ccc14/pcfb/analysis.html
    4PL logistic equation
    sigmoid_2_param: 1.0 / (1.0 + np.exp((x-p)/s)
    (A - d) = 1 in sigmoid_2_param:
    (x/C)**B  - corresponds to np.exp((x-p)/s
    d - determines the vertical position of the graph
    """
    return (A - d) / (1.0 + ((x / C) ** B)) + d


def logLogist_3_param(x, EC50, HS, E_inf):
    """
    Returns y = (1-E_inf)/(1+(np.log10(x)/EC50)**HS) + E_inf
    Python analog for PharmacoGx/R/LogLogisticRegression.R
    https://github.com/bhklab/PharmacoGx/blob/master/R/LogLogisticRegression.R
    E = E_inf + (1 - E_inf)/(1 + (x/EC50)^HS)
    sigmoid_2_param: 1.0 / (1.0 + np.exp((x-p)/s)

    (A - d) = 1 in sigmoid_2_param:
    (np.log10(x)/EC50)**HS  - (in logistic4 (x/C)**B) corresponds to np.exp((x-p)/s

    E_inf - determines the vertical position of the graph /coefficient d, min response in other functions
    """
    return (1 - E_inf) / (1 + (np.log10(x) / EC50) ** HS) + E_inf
