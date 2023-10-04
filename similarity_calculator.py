import numpy as np
from scipy.spatial.distance import cosine
from online_feature_selection import feature_selection_None,feature_selection_original
from normalizer import Normalizer



def get_dimension_weights(fingerprints, state_active_non_active_fingerprints, normalizer=None, state_id=None, feature_selection_method="default"):
    """ Use feature selection methods to weight
    each meta-information feature.
    """
    feature_selection_func = None
    if normalizer == None:
        normalizer = Normalizer(None,None,None)

    if feature_selection_method in ["uniform", "None"]:
        feature_selection_func = feature_selection_None
    if feature_selection_method in ["default", "original"]:
        feature_selection_func = feature_selection_original

    if feature_selection_func is None:
        raise ValueError(
            f"no falid feature selection method {feature_selection_method}")
    return feature_selection_func(fingerprints, state_active_non_active_fingerprints, normalizer, state_id)

def get_cosine_distance(A, B, weighted, weights):
    """ Get cosine distance between vectors A and B.
    Weight vectors first if weighted is set.
    """
    try:
        if not weighted:
            c = cosine(A, B)
        else:
            c = cosine(A, B, w=weights)
    except:
        c = np.nan
    if np.isnan(c):
        c = 0 if ((not np.any(A)) and (not np.any(B))) else 1
    return c


def cosine_similarity(self, current_metainfo, fingerprint_to_compare):
        '''
        if not self.force_lock_weights:
            # Can reuse weights if they exist and were updated this observation, or they haven't changed since last time.
            if self.weights_cache and (not self.normalizer.changed_since_last_weight_calc) and (not self.fingerprint_changed_since_last_weight_calc):
                weights, flat_weight_vector, ignore_flat_weight_vector, sorted_feature_weights = self.weights_cache[
                    0]

            else:
                state_non_active_fingerprints = {k: (v.fingerprint, v.non_active_fingerprints)
                                                 for k, v in self.state_repository.items() if v.fingerprint is not None}
                weights, flat_weight_vector, ignore_flat_weight_vector, sorted_feature_weights = get_dimension_weights(list([state.fingerprint for state in self.state_repository.values(
                ) if state.fingerprint is not None]), state_non_active_fingerprints,  self.normalizer, feature_selection_method=self.feature_selection_method)
                self.weights_cache = (
                    (weights, flat_weight_vector, ignore_flat_weight_vector, sorted_feature_weights), self.ex)
                self.fingerprint_changed_since_last_weight_calc = False
            self.force_locked_weights = (
                weights, flat_weight_vector, ignore_flat_weight_vector, sorted_feature_weights)
        else:
            weights, flat_weight_vector, ignore_flat_weight_vector, sorted_feature_weights = self.force_locked_weights

        self.monitor_feature_selection_weights = sorted_feature_weights

        weights_vec = ignore_flat_weight_vector
        normed_weights = (weights_vec) / (np.max(weights_vec))
        '''

        # Want to check the current fingerprint,
        # and last clean fingerprint
        # and most recent dirty fingerprint (incase there isn't another clean one and recent is new)
        fingerprints_to_check = []
        if fingerprint_to_compare is None:
            fingerprints_to_check.append(
                self.state_repository[state_id].fingerprint)
            for cached_fingerprint in self.state_repository[state_id].fingerprint_cache[::-1]:
                fingerprints_to_check.append(cached_fingerprint)
                if len(fingerprints_to_check) > 5:
                    break
        else:
            fingerprints_to_check = [fingerprint_to_compare]
        similarities = []
        # NOTE: We actually calculate cosine distance, so we return the MINIMUM DISTANCE
        # This is confusing, as you would think if we were really working with similarity it
        # would be maximum!
        # TODO: rename to distance
        for fp in fingerprints_to_check:
            fingerprint_vec, fingerprint_nonorm_vec = self.normalizer.get_flat_vector(fp.fingerprint_values)
            fingerprint_nonorm_vec = fp.flat_ignore_vec
            fingerprint_vec = self.normalizer.norm_flat_vector(
                fingerprint_nonorm_vec)
            similarity = get_cosine_distance(
                stat_vec, fingerprint_vec, True, normed_weights)
            similarities.append(similarity)
        min_similarity = min(similarities)

        return min_similarity

def get_similarity(self, current_metainfo, sim_measure, state=None, fingerprint_to_compare=None, flat_norm_current_metainfo=None, flat_nonorm_current_metainfo=None):
        if sim_measure == "metainfo":
            return cosine_similarity(current_metainfo, state, fingerprint_to_compare=fingerprint_to_compare, flat_norm_current_metainfo=flat_norm_current_metainfo, flat_nonorm_current_metainfo=flat_nonorm_current_metainfo)
        '''
        if self.sim_measure == "sketch":
            return self._sketch_cosine_similarity(current_metainfo, state, fingerprint_to_compare=fingerprint_to_compare, flat_norm_current_metainfo=flat_norm_current_metainfo, flat_nonorm_current_metainfo=flat_nonorm_current_metainfo)
        if self.sim_measure == "histogram":
            return self._histogram_similarity(current_metainfo, state, fingerprint_to_compare=fingerprint_to_compare, flat_norm_current_metainfo=flat_norm_current_metainfo, flat_nonorm_current_metainfo=flat_nonorm_current_metainfo)
        if self.sim_measure == "accuracy":
            return self._accuracy_similarity(current_metainfo, state, fingerprint_to_compare=fingerprint_to_compare)
        raise ValueError("similarity method not set")
        '''