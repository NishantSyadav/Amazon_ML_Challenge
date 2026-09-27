import re
import unicodedata
import pandas as pd


# =========================================================
# Basic text normalization
# =========================================================

def normalize_text(text):
    """
    Basic language-independent normalization.

    - Missing values -> ""
    - Unicode normalization
    - Case folding
    - Whitespace normalization
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Normalize Unicode representation
    text = unicodedata.normalize("NFKC", text)

    # Case-insensitive normalization
    text = text.casefold()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# =========================================================
# Accent normalization
# =========================================================

def remove_accents(text):
    """
    Remove combining accent marks while preserving
    the base Unicode characters.

    Example:
        Énterprises -> Enterprises
        Bóral -> Boral

    Non-Latin scripts are preserved.
    """

    if not text:
        return ""

    text = unicodedata.normalize("NFKD", text)

    result = []

    for char in text:
        # Only remove combining marks
        if unicodedata.category(char) != "Mn":
            result.append(char)

    return "".join(result)


# =========================================================
# Name normalization
# =========================================================

def normalize_name(name):
    """
    Conservative business-name normalization.

    Preserves Unicode scripts and meaningful characters.
    """

    text = normalize_text(name)

    if not text:
        return ""

    # Standardize ampersand
    text = text.replace("&", " and ")

    # Replace punctuation/symbols with spaces.
    # Unicode letters and numbers are preserved.
    cleaned = []

    for char in text:
        category = unicodedata.category(char)

        if category[0] in ("L", "N"):
            cleaned.append(char)

        elif category.startswith("M"):
            # Preserve Unicode combining marks
            cleaned.append(char)

        else:
            cleaned.append(" ")

    text = "".join(cleaned)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_name_ascii(name):
    """
    Accent-normalized name representation.

    This does NOT mean ASCII-only.
    Non-Latin scripts are preserved.
    """

    text = normalize_name(name)

    return remove_accents(text)


# =========================================================
# Address normalization
# =========================================================

def normalize_address(address):
    """
    Conservative address normalization.

    Important:
    Numbers and alphanumeric components are preserved.
    """

    text = normalize_text(address)

    if not text:
        return ""

    cleaned = []

    for char in text:
        category = unicodedata.category(char)

        if category[0] in ("L", "N"):
            cleaned.append(char)

        elif category.startswith("M"):
            cleaned.append(char)

        elif char in "-":
            # Preserve address formats such as:
            # E-3A
            # 1056-1060
            # AF-684
            cleaned.append(char)

        else:
            cleaned.append(" ")

    text = "".join(cleaned)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_address_ascii(address):
    """
    Accent-normalized address representation.
    """

    text = normalize_address(address)

    return remove_accents(text)


# =========================================================
# Token utilities
# =========================================================

def tokenize(text):
    """
    Split normalized text into tokens.
    """

    if not text:
        return []

    return text.split()


def token_set(text):
    """
    Return unique tokens.
    """

    return set(tokenize(text))


# =========================================================
# Numeric utilities
# =========================================================

def extract_numbers(text):
    """
    Extract numeric components from a string.

    Example:
        '3315 Fremont Street'
        -> ['3315']
    """

    if not text:
        return []

    return re.findall(r"\d+", str(text))


# =========================================================
# Missing-value utility
# =========================================================

def has_value(text):
    """
    Return True if the value contains usable text.
    """

    return bool(text and str(text).strip())