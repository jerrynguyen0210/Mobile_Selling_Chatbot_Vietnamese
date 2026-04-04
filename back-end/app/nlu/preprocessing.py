"""Text normalisation for Vietnamese mobile-sales chatbot.

Pipeline (applied in order by :func:`normalize`):

1. Strip & collapse whitespace
2. Lowercase
3. Slang / abbreviation expansion  — loaded from ``slang_map.json``
4. Typo correction via fuzzy matching against :data:`CORRECTION_VOCAB`

**Extending without code changes**

- Slang / abbreviations: edit ``app/nlu/slang_map.json``.
- Typo targets: add the correct spelling to :data:`CORRECTION_VOCAB`.
- Sensitivity: adjust :data:`CORRECTION_CUTOFF` (default 0.82).
"""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Slang / abbreviation map — loaded from slang_map.json
#
# Edit app/nlu/slang_map.json to add or remove entries; no code change
# needed.  The JSON is organised into named sections for readability; the
# loader flattens all sections into a single dict.
# ---------------------------------------------------------------------------

_SLANG_FILE = Path(__file__).with_name("slang_map.json")


def _load_slang_map(path: Path = _SLANG_FILE) -> dict[str, str]:
    """Load and flatten the sectioned slang-map JSON into a single dict."""
    raw: dict[str, dict[str, str] | str] = json.loads(
        path.read_text(encoding="utf-8")
    )
    flat: dict[str, str] = {}
    for value in raw.values():
        if isinstance(value, dict):   # named section → merge entries
            flat.update(value)
        # top-level string values (e.g. "_comment") are ignored
    return flat


SLANG_MAP: dict[str, str] = _load_slang_map()

# ---------------------------------------------------------------------------
# Correction vocabulary
#
# The authoritative list of correctly-spelled terms.  Any token that is
# not an exact match but is "close enough" (≥ CORRECTION_CUTOFF similarity)
# to an entry here will be replaced by that entry automatically.
#
# To support a new brand or product term, add its correct spelling — no
# need to enumerate every possible misspelling.
# ---------------------------------------------------------------------------

CORRECTION_VOCAB: list[str] = [
    # brands
    "samsung", "iphone", "xiaomi", "oppo", "vivo", "realme",
    "nokia", "sony", "huawei", "oneplus", "motorola", "vsmart",
]

# SequenceMatcher ratio threshold in [0, 1].  Raise to reduce false
# positives; lower to catch more distant misspellings.
CORRECTION_CUTOFF: float = 0.82

# Minimum token length before fuzzy correction is attempted.  Short
# tokens (≤ 3 chars) are too ambiguous for reliable fuzzy matching.
_CORRECTION_MIN_LEN: int = 4


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_phrase_map(text: str, mapping: dict[str, str]) -> str:
    """Replace all multi-word keys (keys containing a space) in *text*.

    Applies longest entries first so that ``"bao nhieu"`` is replaced
    before a hypothetical shorter overlapping phrase.
    """
    for source, target in sorted(
        ((k, v) for k, v in mapping.items() if " " in k),
        key=lambda kv: -len(kv[0]),
    ):
        text = text.replace(source, target)
    return text


def _apply_token_map(text: str, mapping: dict[str, str]) -> str:
    """Replace each whitespace-separated token that exactly matches a key.

    Only single-word keys (no spaces) are considered here; multi-word keys
    should already have been handled by :func:`_apply_phrase_map`.
    """
    single: dict[str, str] = {k: v for k, v in mapping.items() if " " not in k}
    return " ".join(single.get(tok, tok) for tok in text.split())


def _correct_typos(text: str) -> str:
    """Fuzzy-correct tokens that are close to an entry in :data:`CORRECTION_VOCAB`.

    Each token is passed to :func:`difflib.get_close_matches` against the
    vocab.  If a single best match is found with ratio ≥
    :data:`CORRECTION_CUTOFF` it replaces the token; otherwise the token is
    left unchanged.  Tokens shorter than :data:`_CORRECTION_MIN_LEN`
    characters are skipped to avoid false positives on short words.
    """
    corrected = []
    for tok in text.split():
        if len(tok) >= _CORRECTION_MIN_LEN:
            matches = difflib.get_close_matches(
                tok, CORRECTION_VOCAB, n=1, cutoff=CORRECTION_CUTOFF
            )
            tok = matches[0] if matches else tok
        corrected.append(tok)
    return " ".join(corrected)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Normalise a raw user message for downstream NLU processing.

    Steps applied in order:

    1. Strip leading/trailing whitespace; collapse internal runs of
       whitespace to a single space.
    2. Lowercase.
    3. Slang / abbreviation expansion — multi-word phrases first, then
       token-level replacements (see :data:`SLANG_MAP`).
    4. Typo correction — fuzzy matching against :data:`CORRECTION_VOCAB`
       using :func:`difflib.get_close_matches`.

    Returns the normalised string.  The function never raises; on an
    empty input it returns an empty string.

    Examples::

        >>> normalize("  Tìm IP 15 giá bao nhieu  ")
        'tìm iphone 15 giá bao nhiêu'

        >>> normalize("dt ss ko co bao hanh ko")
        'điện thoại samsung không có bảo hành không'

        >>> normalize("samsumg galaxy s24")
        'samsung galaxy s24'
    """
    if not text:
        return text

    # 1 — whitespace normalisation
    text = re.sub(r"\s+", " ", text.strip())

    # 2 — lowercase
    text = text.lower()

    # 3 — slang expansion (phrases before tokens)
    text = _apply_phrase_map(text, SLANG_MAP)
    text = _apply_token_map(text, SLANG_MAP)

    # 4 — fuzzy typo correction
    text = _correct_typos(text)

    return text
