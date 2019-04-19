import matplotlib.pyplot as plt
import pickle
import numpy

SVM = "svm"
ARIMA = "arima"
BASE = "base"


def get_mean(list, is_base=False):
    mean = 0.0
    for mse in list:
        if is_base:
            mean += mse[0]
        else:
            mean += mse
    mean /= len(list)
    return mean


def get_variance(list, mean, is_base=False):
    variance = 0.0
    for mse in list:
        if is_base:
            variance += (mean - mse[0]) ** 2
        else:
            variance += (mean - mse) ** 2
    variance /= len(list)
    return variance


def get_std_dev(variance):
    return numpy.sqrt(variance)


def get_max(list, is_base=False):
    maxi = 0.0
    for mse in list:
        if is_base:
            if mse[0] > maxi:
                maxi = mse[0]
        else:
            if mse > maxi:
                maxi = mse
    return maxi


def get_min(list, is_base=False):
    mini = 100000.0
    for mse in list:
        if is_base:
            if mse[0] < mini:
                mini = mse[0]
        else:
            if mse < mini:
                mini = mse
    return mini


def get_median(list, is_base=False):
    if is_base:
        list = sorted(list, key=lambda x: x[0])
    else:
        list = sorted(list)
    median = int(len(list) / 2) + 1
    if is_base:
        median_element = list[median][0]
    else:
        median_element = list[median]
    return median_element


arima = pickle.load(open("mse_all_arima.pkl", "rb"))
svm = pickle.load(open("mse_svm_per_patient_ascending.pkl", "rb"))
base = pickle.load(open("mse_base.pkl", "rb"))

del svm[19]
del svm[17]

del base[19]
del base[17]

mean = {}
mean[BASE] = get_mean(base, True)
mean[ARIMA] = get_mean(arima)
mean[SVM] = get_mean(svm)

variance = {}
variance[BASE] = get_variance(base, mean[BASE], True)
variance[ARIMA] = get_variance(arima, mean[ARIMA])
variance[SVM] = get_variance(svm, mean[SVM])

std_dev = {}
std_dev[BASE] = get_std_dev(variance[BASE])
std_dev[ARIMA] = get_std_dev(variance[ARIMA])
std_dev[SVM] = get_std_dev(variance[SVM])

mini = {}
mini[BASE] = get_min(base, True)
mini[ARIMA] = get_min(arima)
mini[SVM] = get_min(svm)

median = {}
median[BASE] = get_median(base, True)
median[ARIMA] = get_median(arima)
median[SVM] = get_median(svm)
print(median)
