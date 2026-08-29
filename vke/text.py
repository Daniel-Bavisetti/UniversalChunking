"""Text utilities shared across the adapter boundary.

`tokenize` is used by the video-specific signal code AND by modality-neutral
consumers (enrichment, retrieval). It lives here rather than in `signals.py` so
that downstream modules never have to import a video module to handle text -
which is the rule that lets a future PDF pipeline reuse them unchanged.
"""

from __future__ import annotations

import re

# Small, deliberate stopword list. TextTiling and TF-IDF both work on content-word
# overlap, so function words are pure noise.
STOPWORDS = frozenset("""
a about above after again against all am an and any are aren't as at be because
been before being below between both but by can cannot could couldn't did didn't
do does doesn't doing don't down during each few for from further had hadn't has
hasn't have haven't having he her here hers herself him himself his how i if in
into is isn't it its itself let's me more most mustn't my myself no nor not of
off on once only or other ought our ours ourselves out over own same shan't she
should shouldn't so some such than that the their theirs them themselves then
there these they this those through to too under until up very was wasn't we
were weren't what when where which while who whom why with won't would wouldn't
you your yours yourself yourselves will just now also can't it's that's we're
they're i'm you're he's she's there's here's what's let us going get got make
made take taken see seen say said thing things way ways lot lots
""".split())

_WORD = re.compile(r"[a-z][a-z'\-]+")


def tokenize(text: str) -> list[str]:
    """Lowercased content words, in order. Stopwords and short tokens dropped."""
    return [w for w in _WORD.findall(text.lower())
            if w not in STOPWORDS and len(w) > 2]
