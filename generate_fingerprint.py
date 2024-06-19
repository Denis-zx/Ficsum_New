import numpy as np
import shap
import scipy.stats

from PyEMD import EMD
from antropy import perm_entropy
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import acf, pacf

def make_shap_model(model,X_train):
    masker = shap.maskers.Independent(data = X_train)
    return shap.explainers.Linear(model,masker=masker)

def turningpoints(lst):
    dx = np.diff(lst)
    return np.sum(dx[1:] * dx[:-1] < 0)

def flaten_fingerprint(finger):
    '''
    Fingerprint input is list(dict())
    Flaten each fingerprint in the pool to a single array
    '''
    features = finger[0].keys()
    flaten_fingerprint = []
    for sourse in finger:
        flaten_fingerprint+=[sourse[feature] for feature in features]
    flaten_fingerprint = np.array(flaten_fingerprint)
    flaten_fingerprint[np.isnan(flaten_fingerprint)] = 0
    return flaten_fingerprint

def fingerprint_percolumn(timeseries,FI= None,ignore_features=[]):
    """ 
    Calculates a set of statistics for a given timeseries(columns).
    """
    stats = {}
    with np.errstate(divide='ignore', invalid='ignore'):
        if 'IMF' not in ignore_features:
            emd = EMD(max_imf=2, spline_kind='slinear')
            IMFs = emd(np.array(timeseries), max_imf=2)
            for i, imf in enumerate(IMFs):
                #print(imf)
                if f"IMF_{i}" not in ignore_features:
                    stats[f"IMF_{i}"] = perm_entropy(imf)
            for i in range(3):
                if f"IMF_{i}" not in stats and f"IMF_{i}" not in ignore_features:
                    stats[f"IMF_{i}"] = 0
    
        if 'mean' not in ignore_features:
            stats["mean"] = np.mean(timeseries)
        if 'stdev' not in ignore_features:
            stats["stdev"] = np.std(timeseries, ddof=1)
        if 'skew' not in ignore_features:
            stats["skew"] = scipy.stats.skew(timeseries)
        if 'kurtosis' not in ignore_features:
            stats['kurtosis'] = scipy.stats.kurtosis(timeseries)
        if 'turning_point_rate' not in ignore_features:
            tp = int(turningpoints(timeseries))
            tp_rate = tp / len(timeseries)
            stats['turning_point_rate'] = tp_rate

        if 'acf' not in ignore_features:
            acf_vals = acf(timeseries, nlags=3, fft=True)
            for i, v in enumerate(acf_vals):
                if i == 0:
                    continue
                if i > 2:
                    break
                if f"acf_{i}" not in ignore_features:
                    stats[f"acf_{i}"] = v if not np.isnan(v) else -1

        if 'pacf' not in ignore_features:
            try:
                pacf_vals = pacf(timeseries, nlags=3)
            except:
                pacf_vals = [-1 for x in range(6)]
            for i, v in enumerate(pacf_vals):
                if i == 0:
                    continue
                if i > 2:
                    break
                if f"pacf_{i}" not in ignore_features:
                    stats[f"pacf_{i}"] = v if not np.isnan(v) else -1

        if 'MI' not in ignore_features:
            if len(timeseries) > 4:
                current = np.array(timeseries)
                previous = np.roll(current, -1)
                current = current[:-1]
                previous = previous[:-1]
                X = np.array(current).reshape(-1, 1)
                # Setting the random state is mostly for testing.
                # It can induce randomness in MI which is weird for paired
                # testing, getting different results with the same feature vec.
                MI = mutual_info_regression(
                    X=X, y=previous, random_state=42, copy=False)[0]
            else:
                MI = 0
            stats["MI"] = MI

        if 'FI' not in ignore_features:
            stats["FI"] = FI if FI is not None else 0

    '''
    # Return lowerbound and range of current feature
    cur_lowerbound = np.min(timeseries)
    cur_range = np.max(timeseries)-cur_lowerbound
    '''

    return stats

def fingerprint_generator(expredicted_datastream):
    fingerprint = []
    lowerbounds = []
    feature_ranges = []
    for col in expredicted_datastream.columns:
        stats = fingerprint_percolumn(expredicted_datastream[col])
        fingerprint.append(stats)
        '''
        stats,cur_lowerbound,cur_range = fingerprint_percolumn(expredicted_datastream[col])
        fingerprint.append(stats)
        lowerbounds.append(cur_lowerbound)
        feature_ranges.append(cur_range)
        '''
    return fingerprint