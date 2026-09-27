import re
import numpy as np
from rapidfuzz import fuzz

from normalize import (
    normalize_name,
    normalize_name_ascii,
    normalize_address,
    normalize_address_ascii,
    extract_numbers,
)


# =========================================================
# Utility functions
# =========================================================

def safe_ratio(value):
    """
    Convert a value to a safe float.
    """

    if value is None:
        return 0.0

    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def length_ratio(text1, text2):
    """
    Ratio of shorter string length to longer string length.
    """

    len1 = len(text1)
    len2 = len(text2)

    if len1 == 0 and len2 == 0:
        return 1.0

    if len1 == 0 or len2 == 0:
        return 0.0

    return min(len1, len2) / max(len1, len2)


def token_jaccard(text1, text2):
    """
    Jaccard similarity between two token sets.
    """

    tokens1 = set(text1.split())
    tokens2 = set(text2.split())

    if not tokens1 and not tokens2:
        return 1.0

    if not tokens1 or not tokens2:
        return 0.0

    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)

    return intersection / union


def number_overlap(text1, text2):
    """
    Measure overlap between numeric components.

    Example:

        '3315 Fremont Street'
        '3315 Fremont St'

        -> 1.0
    """

    nums1 = set(extract_numbers(text1))
    nums2 = set(extract_numbers(text2))

    if not nums1 and not nums2:
        return 1.0

    if not nums1 or not nums2:
        return 0.0

    intersection = len(nums1 & nums2)

    return intersection / max(len(nums1), len(nums2))


# =========================================================
# Name features
# =========================================================

def name_features(name1, name2):

    n1 = normalize_name(name1)
    n2 = normalize_name(name2)

    a1 = normalize_name_ascii(name1)
    a2 = normalize_name_ascii(name2)

    features = {}

    # Missing flags
    features["name1_missing"] = int(n1 == "")
    features["name2_missing"] = int(n2 == "")

    # Exact matching
    features["name_exact"] = int(n1 == n2)
    features["name_ascii_exact"] = int(a1 == a2)

    # Fuzzy similarity
    features["name_ratio"] = fuzz.ratio(n1, n2) / 100.0
    features["name_partial_ratio"] = fuzz.partial_ratio(n1, n2) / 100.0
    features["name_token_sort_ratio"] = (
        fuzz.token_sort_ratio(n1, n2) / 100.0
    )
    features["name_token_set_ratio"] = (
        fuzz.token_set_ratio(n1, n2) / 100.0
    )

    # Token similarity
    features["name_token_jaccard"] = token_jaccard(n1, n2)

    # Length similarity
    features["name_length_ratio"] = length_ratio(n1, n2)

    return features


# =========================================================
# Address features
# =========================================================

def address_features(address1, address2):

    a1 = normalize_address(address1)
    a2 = normalize_address(address2)

    aa1 = normalize_address_ascii(address1)
    aa2 = normalize_address_ascii(address2)

    features = {}

    # Missing flags
    features["address1_missing"] = int(a1 == "")
    features["address2_missing"] = int(a2 == "")

    # Exact matching
    features["address_exact"] = int(a1 == a2)
    features["address_ascii_exact"] = int(aa1 == aa2)

    # Fuzzy similarity
    features["address_ratio"] = fuzz.ratio(a1, a2) / 100.0
    features["address_partial_ratio"] = (
        fuzz.partial_ratio(a1, a2) / 100.0
    )

    features["address_token_sort_ratio"] = (
        fuzz.token_sort_ratio(a1, a2) / 100.0
    )

    features["address_token_set_ratio"] = (
        fuzz.token_set_ratio(a1, a2) / 100.0
    )

    # Token similarity
    features["address_token_jaccard"] = token_jaccard(a1, a2)

    # Numeric overlap
    features["address_number_overlap"] = number_overlap(a1, a2)

    # Length similarity
    features["address_length_ratio"] = length_ratio(a1, a2)

    return features


# =========================================================
# Metadata features
# =========================================================

def metadata_features(
    country1,
    country2,
    candidate_source=None,
    retrieval_score=None,
    retrieval_rank=None,
):

    c1 = "" if country1 is None else str(country1).strip().casefold()
    c2 = "" if country2 is None else str(country2).strip().casefold()

    features = {}

    # Country
    features["same_country"] = int(
        c1 != "" and c2 != "" and c1 == c2
    )

    # Candidate source
    features["candidate_is_s2"] = int(candidate_source == "S2")
    features["candidate_is_s3"] = int(candidate_source == "S3")

    # Retrieval information
    features["retrieval_score"] = safe_ratio(retrieval_score)

    if retrieval_rank is None:
        features["retrieval_rank"] = 0.0
    else:
        features["retrieval_rank"] = safe_ratio(retrieval_rank)

    return features


# =========================================================
# Complete pair feature function
# =========================================================

def build_pair_features(
    name1,
    address1,
    country1,
    name2,
    address2,
    country2,
    candidate_source=None,
    retrieval_score=None,
    retrieval_rank=None,
):
    """
    Build all features for one S1-candidate pair.
    """

    features = {}

    # Name
    features.update(
        name_features(name1, name2)
    )

    # Address
    features.update(
        address_features(address1, address2)
    )

    # Metadata
    features.update(
        metadata_features(
            country1,
            country2,
            candidate_source,
            retrieval_score,
            retrieval_rank,
        )
    )

    return features