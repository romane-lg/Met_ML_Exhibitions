from __future__ import annotations

import re
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


@lru_cache(maxsize=1)
def get_stopwords() -> set[str]:
    try:
        return set(stopwords.words("english"))
    except LookupError:
        try:
            nltk.download("stopwords", quiet=True)
            return set(stopwords.words("english"))
        except Exception:
            return set()


@lru_cache(maxsize=1)
def get_lemmatizer() -> WordNetLemmatizer:
    try:
        return WordNetLemmatizer()
    except LookupError:
        try:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        except Exception:
            pass
        return WordNetLemmatizer()


def tokenize_text(
    text: str,
    *,
    stop_words: set[str] | None = None,
    lemmatizer: WordNetLemmatizer | None = None,
    min_len: int = 2,
    keep_numbers: bool = False,
) -> list[str]:
    stop_words = stop_words if stop_words is not None else get_stopwords()
    lemmatizer = lemmatizer if lemmatizer is not None else get_lemmatizer()
    normalized = re.sub(r"[^a-z0-9\s]", " ", str(text).lower())
    tokens = normalized.split()
    return [
        lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok and tok not in stop_words and len(tok) > min_len and (keep_numbers or not tok.isdigit())
    ]
