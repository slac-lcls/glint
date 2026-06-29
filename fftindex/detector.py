"""Learned peakfinder: re-rank candidate volume peaks by a trained classifier.

Drops into index_shot's peakfinder slot. It runs the classical finder to get a
generous candidate pool, scores each candidate with peak_features + the trained
model, and returns the candidates ordered by predicted P(true axis) -- so true
axes survive into search_basis's top-k even when jitter/spurious have buried them
in raw amplitude. The model's probability replaces "amp" in the (vecs, amps) tuple.
"""

from __future__ import annotations

import numpy as np

from .features import peak_features
from .peakfind import find_peaks_classical


class LearnedPeakFinder:
    def __init__(self, clf, pool=40):
        self.clf = clf          # any sklearn classifier with predict_proba
        self.pool = pool        # how many classical candidates to re-rank

    def __call__(self, vol, xcoords, g=None, qmax=None, min_len=3.0):
        vecs, amps = find_peaks_classical(vol, xcoords, min_len=min_len,
                                          max_peaks=self.pool)
        if len(vecs) == 0 or g is None:
            return vecs, amps
        prob = self.clf.predict_proba(peak_features(g, vecs, qmax))[:, 1]
        order = np.argsort(prob)[::-1]
        return vecs[order], prob[order]
