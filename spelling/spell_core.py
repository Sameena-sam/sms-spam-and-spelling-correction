"""
spell_core.py
=============
Spelling-correction engine for the NLP assignment (Part A, Half 1 - 30 marks).

Scope (per the module requirement): the system corrects **spelling errors only**
- i.e. NON-WORD errors, where the typed token does not exist in the dictionary
(e.g. "compnay" -> "company").  It does **not** perform context / real-word
correction (it never changes a correctly-spelled word based on its neighbours).

The dictionary is DOMAIN-SPECIFIC: it is built from a single specialised corpus.
The default domain is **business / financial news** - the NLTK *Reuters* corpus
(~1.3 million word tokens), which satisfies the ">= 100,000 words in a sciences
field" requirement (business science) and is large enough to give reliable
coverage of ordinary business vocabulary.  A local plain-text corpus for any
other domain can be supplied via `path=`.

NLP techniques implemented:
  * Minimum Edit Distance   - Levenshtein, with optional Damerau transposition
                              (used to generate and score candidate corrections).
  * Unigram frequency model - the noisy-channel prior that ranks candidates.
  * Bigram language model    - built from the corpus and available for analysis
                              (see `bigram_top`); demonstrated in the report but
                              NOT used for correction, as context correction is
                              out of scope.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

# A "word" = letters, optionally with an internal apostrophe (don't, it's).
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def tokenize(text: str):
    """Lower-case and split text into word tokens."""
    return WORD_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# 1. Minimum edit distance  (required technique)
# ---------------------------------------------------------------------------
def edit_distance(a: str, b: str, transpositions: bool = True) -> int:
    """Minimum edit distance between strings `a` and `b`.

    Insertion, deletion and substitution each cost 1.
    transpositions=True  -> Damerau-Levenshtein (an adjacent swap costs 1),
    transpositions=False -> plain Levenshtein.
    """
    la, lb = len(a), len(b)
    prev2 = None                       # row i-2  (needed for transpositions)
    prev = list(range(lb + 1))         # row i-1
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1,          # deletion
                         cur[j - 1] + 1,       # insertion
                         prev[j - 1] + cost)   # substitution
            if (transpositions and i > 1 and j > 1
                    and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]):
                cur[j] = min(cur[j], prev2[j - 2] + 1)   # transposition
        prev2, prev = prev, cur
    return prev[lb]


# ---------------------------------------------------------------------------
# 2. Candidate generation  (Norvig-style edits at distance 1 and 2)
# ---------------------------------------------------------------------------
def _edits1(word: str):
    """All strings one edit away from `word`."""
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in ALPHABET]
    inserts = [L + c + R for L, R in splits for c in ALPHABET]
    return set(deletes + transposes + replaces + inserts)


def _edits2(word: str):
    """All strings two edits away from `word`."""
    return {e2 for e1 in _edits1(word) for e2 in _edits1(e1)}


# ---------------------------------------------------------------------------
# 3. The spell checker
# ---------------------------------------------------------------------------
class SpellChecker:
    """Builds a unigram (and, for demonstration, a bigram) language model from a
    domain corpus and corrects non-word spelling errors with the noisy channel:
    candidates come from small edit distances, ranked by corpus word frequency.
    """

    def __init__(self, tokens, smoothing: float = 1.0, min_count: int = 5):
        # Unigram model = corpus word frequencies (the noisy-channel prior).
        self.unigram = Counter(tokens)
        self.N = sum(self.unigram.values())
        # The DICTIONARY is domain-specific: words seen at least `min_count`
        # times in the corpus. Pruning very rare tokens removes typos / OCR
        # noise so the checker does not accept corpus misspellings as valid.
        self.min_count = min_count
        self.vocab = {w for w, c in self.unigram.items() if c >= min_count}
        self.V = len(self.vocab)

        # Bigram model: counts of (previous_word -> word). Built from the corpus
        # and used for analysis/reporting only (see bigram_top); NOT used for
        # correction, because context correction is out of scope.
        self.bigram = defaultdict(Counter)
        prev = None
        for w in tokens:
            if prev is not None:
                self.bigram[prev][w] += 1
            prev = w

        self.k = smoothing                       # add-k smoothing constant
        self.sorted_words = sorted(self.vocab)   # for the GUI dictionary list

    # ---- probability models -------------------------------------------------
    def p_unigram(self, w: str) -> float:
        """Smoothed P(w)."""
        return (self.unigram.get(w, 0) + self.k) / (self.N + self.k * self.V)

    def p_bigram(self, w: str, given: str) -> float:
        """Smoothed P(w | given) with add-k smoothing (for bigram analysis)."""
        ctx = self.bigram.get(given)
        num = (ctx.get(w, 0) if ctx else 0) + self.k
        denom = self.unigram.get(given, 0) + self.k * self.V
        return num / denom

    def bigram_top(self, prev: str, n: int = 8):
        """Most frequent words that follow `prev` in the corpus.
        Used in the report to demonstrate the bigram language model."""
        ctx = self.bigram.get(prev.lower())
        return ctx.most_common(n) if ctx else []

    # ---- helpers ------------------------------------------------------------
    def _known(self, words):
        """Keep only the words that exist in the dictionary."""
        return {w for w in words if w in self.vocab}

    def is_nonword(self, word: str) -> bool:
        """True if `word` is not in the domain dictionary (a spelling error)."""
        return word.lower() not in self.vocab

    # ---- non-word spelling correction --------------------------------------
    def suggest_nonword(self, word: str, n: int = 5):
        """Ranked corrections for a misspelled (non-)word.

        Returns a list of (candidate, edit_distance). Ranking follows the
        noisy-channel idea: a small edit distance dominates, and the corpus
        unigram frequency P(candidate) breaks ties between equally close words.
        No surrounding-word context is used - this corrects spelling only.
        """
        word = word.lower()
        cands = self._known(_edits1(word)) or self._known(_edits2(word))
        if not cands:
            return []
        scored = []
        for c in cands:
            d = edit_distance(word, c)
            score = math.log(self.p_unigram(c)) - 4.0 * d   # distance dominates
            scored.append((c, d, score))
        scored.sort(key=lambda t: t[2], reverse=True)
        return [(c, d) for c, d, _ in scored[:n]]

    # ---- whole-text check (used by the GUI) --------------------------------
    def check_text(self, text: str):
        """Scan `text` and return one issue per misspelled (non-word) token:
            {start, end, word, type: 'nonword', suggestions: [(word, distance)]}
        `start`/`end` are character offsets into the original text.
        """
        issues = []
        for m in WORD_RE.finditer(text.lower()):    # offsets match the original
            w = m.group()
            if w not in self.vocab:
                issues.append(dict(start=m.start(), end=m.end(), word=w,
                                   type="nonword",
                                   suggestions=self.suggest_nonword(w)))
        return issues

    # ---- dictionary search (used by the GUI) -------------------------------
    def search_dictionary(self, query: str, limit: int = 500):
        """Return up to `limit` dictionary words matching `query`
        (prefix matches first, then any substring match)."""
        q = query.strip().lower()
        if not q:
            return self.sorted_words[:limit]
        prefix = [w for w in self.sorted_words if w.startswith(q)]
        contains = [w for w in self.sorted_words if q in w and not w.startswith(q)]
        return (prefix + contains)[:limit]


# ---------------------------------------------------------------------------
# 4. Corpus loading
# ---------------------------------------------------------------------------
def load_corpus_tokens(path: str | None = None):
    """Load corpus tokens for the chosen domain.

    * If `path` is given, read that UTF-8 text file (plug in any domain corpus).
    * Otherwise download/use the NLTK *Reuters* corpus (business / financial
      news), keeping only alphabetic word tokens.
    """
    if path:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return tokenize(f.read())

    import nltk
    try:
        from nltk.corpus import reuters
        reuters.fileids()                     # triggers LookupError if missing
    except LookupError:
        nltk.download("reuters")
        from nltk.corpus import reuters
    return [w.lower() for w in reuters.words() if w.isalpha()]


def build_checker(path: str | None = None, smoothing: float = 1.0,
                  min_count: int = 5):
    """Convenience: load the domain corpus and return (SpellChecker, tokens).

    `min_count` prunes rare corpus tokens (typos / noise) from the dictionary.
    """
    tokens = load_corpus_tokens(path)
    return SpellChecker(tokens, smoothing=smoothing, min_count=min_count), tokens


# ---------------------------------------------------------------------------
# Quick self-test from the command line:  python spell_core.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading corpus (NLTK Reuters - business/financial news)...")
    checker, tokens = build_checker()
    print(f"Corpus size : {len(tokens):,} tokens")
    print(f"Dictionary  : {len(checker.vocab):,} domain words")
    print("Non-word 'compnay'   ->", checker.suggest_nonword("compnay"))
    print("Non-word 'finacial'  ->", checker.suggest_nonword("finacial"))
    print("Non-word 'managment' ->", checker.suggest_nonword("managment"))
    print("Bigram: top words after 'interest' ->", checker.bigram_top("interest", 5))
