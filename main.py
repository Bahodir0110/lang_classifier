import re
import sys
from collections import defaultdict
import numpy as np
import fasttext

# 1. Fast NumPy 2.x compatibility patch (avoids **kwargs dict overhead per chunk)
_orig_array = np.array
def _safe_array(obj, dtype=None, copy=True):
    return _orig_array(obj, dtype=dtype)
np.array = _safe_array

# Suppress harmless fastText loading warning
fasttext.FastText.eprint = lambda *args, **kwargs: None

# 2. Pre-compiled C-level regexes for script runs and word extraction
# Splits text only when switching between Cyrillic and Latin (ignores commas/numbers)
SCRIPT_RUN_RE = re.compile(
    r"[\u0400-\u052F][^a-zA-Z\u00C0-\u024F]*|"
    r"[a-zA-Z\u00C0-\u024F][^\u0400-\u052F]*"
)
WORD_RE = re.compile(r"[a-zA-Z\u00C0-\u024F\u0400-\u052Fʻʼ'’`]+")

# 3. Load model
model = fasttext.load_model("lang_classifier.ftz")


def predict_composition(text: str, batch_size: int = 5) -> dict[str, float]:
    """
    Splits text into script-aware 5-word batches, runs a single C++ batch
    prediction, and returns the language composition percentages.
    """
    text = text.strip()
    if not text:
        return {}

    batches = []
    batch_weights = []

    # Group contiguous same-script runs first, then slice into 5-word batches
    for match in SCRIPT_RUN_RE.finditer(text):
        words = WORD_RE.findall(match.group(0))
        if not words:
            continue

        for i in range(0, len(words), batch_size):
            chunk = words[i : i + batch_size]
            # Merge 1-2 trailing orphan words into the previous same-script batch
            # so single words aren't classified without morphological context
            if len(chunk) < 3 and i > 0:
                batches[-1] = f"{batches[-1]} {' '.join(chunk)}"
                batch_weights[-1] += len(chunk)
            else:
                batches.append(" ".join(chunk))
                batch_weights.append(len(chunk))

    if not batches:
        return {}

    # Fast path: single batch skips list-of-strings overhead
    if len(batches) == 1:
        labels, _ = model.predict(batches[0], k=1)
        return {labels[0].replace("__label__", ""): 100.0}

    # Single C++ batch inference call across all 5-word chunks
    pred_labels, _ = model.predict(batches, k=1)

    lang_counts = defaultdict(int)
    total_words = sum(batch_weights)

    for weight, lbl_list in zip(batch_weights, pred_labels):
        lang = lbl_list[0].replace("__label__", "")
        lang_counts[lang] += weight

    return {
        lang: round((count / total_words) * 100, 1)
        for lang, count in sorted(lang_counts.items(), key=lambda x: x[1], reverse=True)
    }


def predict(text: str, batch_size: int = 5):
    """
    Drop-in replacement for predict(text):
    Returns (dominant_label, dominant_percentage, full_composition_dict).
    """
    composition = predict_composition(text, batch_size=batch_size)
    if not composition:
        return "unknown", 0.0, {}

    top_label, top_share = next(iter(composition.items()))
    return top_label, top_share, composition


