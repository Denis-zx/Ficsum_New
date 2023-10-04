import numpy as np

def get_weight_sd(classifier):
    weights = []
    cur_fingerprint = classifier.cur_fingerprint
    for idx in range (len(cur_fingerprint)):
        weights.append(0.01/max(cur_fingerprint[idx]["stdev"], 0.01))
    return weights

def get_weight_dmi(repo):
    weight_dmi = []
    num_features = len(repo[0][0])
    for feature_idx in range(num_features):
        weight_dmi.append( get_weight_Vsmi([fingerprint[0][feature_idx] for fingerprint in repo], repo[0][2][feature_idx]) )
    return weight_dmi

def get_weight_intra_sd_per_feature(classifier,feature):
    '''
    Allocated weight according to the std of each feature
    '''
    fingerprints = classifier.fingerprint_pool
    
    standard_deviations = [finger[feature]["stdev"] for finger in fingerprints]
    mean_stdev = np.mean(standard_deviations)
    weight = 0.01 / max(mean_stdev, 0.01)
    return weight

def get_weight_intra (classifier):
    weights = []
    cur_fingerprint = classifier.cur_fingerprint
    for idx in range (len(cur_fingerprint)):
        weights.append(get_weight_intra_sd_per_feature(classifier,idx))
    return weights


def get_weight_Vsmi(features_stat,scaling_factor):
    feature_means = np.array([stat["mean"] for stat in features_stat])
    feature_stds = np.array([stat["stdev"] for stat in features_stat])
    
    std_feature_means = np.std(feature_means)
    max_stdev = (np.max(feature_stds) / scaling_factor) #divide by scaling factor
    rev_max_stdev = 0.01 / max(max_stdev, 0.01)
    weight_Vsmi= std_feature_means * rev_max_stdev
    return weight_Vsmi
        

