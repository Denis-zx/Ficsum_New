import numpy as np
from scipy.spatial.distance import cosine

def get_cosine_distance(A, B, weights,weighted=True):
    """ 
    Get cosine distance between vectors A and B.
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


