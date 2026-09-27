"""Production features: numerically identical to the saved V3.1 training features."""
from functools import lru_cache
import numpy as np
from rapidfuzz import fuzz
from src.baseline_features import normalize_text, length_ratio

FEATURE_COLUMNS = [
    'name_ratio', 'name_partial_ratio', 'name_token_sort', 'name_token_set',
    'name_exact', 'name_length_ratio', 'address_ratio', 'address_partial_ratio',
    'address_token_sort', 'address_token_set', 'address_exact', 'address_length_ratio',
    'same_country', 'max_name_similarity', 'max_address_similarity', 'combined_similarity',
]
_normalize = lru_cache(maxsize=100_000)(normalize_text)


def feature_matrix(rows, normalized=False):
    """Rows are (S1 name, address, country, candidate name, address, country)."""
    result = np.empty((len(rows), 16), dtype=np.float64)
    scorers = (fuzz.ratio, fuzz.partial_ratio, fuzz.token_sort_ratio, fuzz.token_set_ratio)
    for i, (n1, a1, c1, n2, a2, c2) in enumerate(rows):
        if not normalized:
            n1, a1, n2, a2 = map(_normalize, (n1, a1, n2, a2))
        for offset, a, b in ((0, n1, n2), (6, a1, a2)):
            if a and b:
                values = [scorer(a,b)/100.0 for scorer in scorers]
                result[i,offset:offset+6] = values + [int(a==b),length_ratio(a,b)]
            else:
                result[i,offset:offset+6] = 0
        result[i,12] = int(c1 == c2)
        result[i,13] = max(result[i,:4])
        result[i,14] = max(result[i,6:10])
        result[i,15] = 0.6*result[i,13] + 0.4*result[i,14]
    return result.astype(np.float32)
