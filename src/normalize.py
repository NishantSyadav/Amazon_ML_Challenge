# quick and dirty text cleanup for business names/addresses.
# this is just for blocking (matching TF-IDF text + exact-match keys) -
# Nishant's feature-engineering normalization for the actual model can be
# stricter/different, doesn't need to match this exactly.

import re
import unicodedata

LEGAL_SUFFIXES = {
    "pvt": "private", "ltd": "limited", "corp": "corporation",
    "inc": "incorporated", "co": "company", "llc": "llc",
    "llp": "llp", "plc": "plc",
}

ADDRESS_ABBR = {
    "rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard",
    "apt": "apartment", "no": "number", "flr": "floor", "bldg": "building",
    "dist": "district", "twp": "township",
}

PUNCT_RE = re.compile(r"[^\w\s]")
WS_RE = re.compile(r"\s+")


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def expand(tokens, mapping):
    return [mapping.get(t, t) for t in tokens]


def normalize_name(name):
    if not isinstance(name, str) or not name:
        return ""
    s = strip_accents(name.lower()).replace("&", " and ")
    s = PUNCT_RE.sub(" ", s)
    tokens = expand(s.split(), LEGAL_SUFFIXES)
    return WS_RE.sub(" ", " ".join(tokens)).strip()


def normalize_address(address):
    if not isinstance(address, str) or not address:
        return ""
    s = strip_accents(address.lower())
    s = PUNCT_RE.sub(" ", s)
    tokens = expand(s.split(), ADDRESS_ABBR)
    return WS_RE.sub(" ", " ".join(tokens)).strip()


def extract_numbers(text):
    """pulls house/PIN/unit numbers out - language independent, works fine on
    French addresses too even though we've never seen one in training."""
    if not isinstance(text, str):
        return []
    return re.findall(r"\d+", text)
