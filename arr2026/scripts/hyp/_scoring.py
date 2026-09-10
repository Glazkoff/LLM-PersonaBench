r"""Scoring rules for 5-level ordinal questionnaire forecasts.

Every rule maps a forecast simplex P (n, J, 5) and observed answers Y (n, J)
to a per-(respondent, item) score, so selection and evaluation read the same
object. Orientation is normalised here: every rule is returned LOWER-IS-BETTER,
including the ones whose natural form is a reward, so a selector is always an
argmin and cannot silently invert.

The families differ in what they reward:

  ordinal + proper    rps          the paper's S_1/2; penalises a four-category
                                   error more than a one-category error
  categorical proper  logloss      local, ignores order, unbounded
                      brier        bounded, ignores order
                      spherical    bounded, ignores order
  improper            s0           the audited metric, S_0; constant-optimal
                      ev_mae       point prediction by posterior mean
                      ev_rmse      point prediction, quadratic
                      mode_acc     argmax accuracy, discards the distribution
                      wass_point   Wasserstein-1 to the point mass at y

"Proper" is asserted by test in test_scoring.py, not by this comment: for each
rule we check numerically that reporting the true distribution beats reporting
a point mass at its median, which is exactly the property S_0 fails.
"""
from __future__ import annotations

import numpy as np

LEVELS = np.arange(1, 6, dtype=float)
THRESH = [1, 2, 3, 4]


def _cdf(P):
    return np.cumsum(P, axis=-1)[..., :4]


def _ind(Y):
    return np.stack([(Y <= t).astype(float) for t in THRESH], axis=-1)


def rps(P, Y):
    """Ranked probability score == the paper's S_1/2, normalised to [0,1]."""
    return np.nansum((_cdf(P) - _ind(Y)) ** 2, axis=-1) / 4.0


def logloss(P, Y):
    idx = np.clip(np.nan_to_num(Y, nan=1).astype(int) - 1, 0, 4)
    p = np.take_along_axis(P, idx[..., None], axis=-1)[..., 0]
    # A genuine zero is infinitely wrong; only floor tiny-but-real values so
    # they stay finite in float. Never clip a true zero into a finite number.
    return np.where(p > 0, -np.log(np.clip(p, 1e-300, None)), np.inf)


def brier(P, Y):
    idx = np.clip(np.nan_to_num(Y, nan=1).astype(int) - 1, 0, 4)
    oh = np.zeros_like(P)
    np.put_along_axis(oh, idx[..., None], 1.0, axis=-1)
    return np.sum((P - oh) ** 2, axis=-1)


def spherical(P, Y):
    idx = np.clip(np.nan_to_num(Y, nan=1).astype(int) - 1, 0, 4)
    p = np.take_along_axis(P, idx[..., None], axis=-1)[..., 0]
    nrm = np.sqrt(np.sum(P ** 2, axis=-1))
    return -p / np.where(nrm > 0, nrm, 1.0)


def s0(P, Y):
    """The audited metric, negated so lower is better. S_0 = 1 - E|X-y|/4."""
    d = np.abs(LEVELS - Y[..., None]) / 4.0
    return -(1.0 - np.sum(P * d, axis=-1))


def ev_mae(P, Y):
    return np.abs(np.sum(P * LEVELS, axis=-1) - Y) / 4.0


def ev_rmse(P, Y):
    return ((np.sum(P * LEVELS, axis=-1) - Y) / 4.0) ** 2


def mode_acc(P, Y):
    """Negated argmax accuracy. Discards everything but the mode."""
    return -(np.argmax(P, axis=-1) + 1.0 == Y).astype(float)


def wass_point(P, Y):
    """Wasserstein-1 between the forecast and the point mass at y == E|X-y|."""
    return np.sum(np.abs(_cdf(P) - _ind(Y)), axis=-1) / 4.0


RULES = {
    "rps": rps, "logloss": logloss, "brier": brier, "spherical": spherical,
    "s0": s0, "ev_mae": ev_mae, "ev_rmse": ev_rmse, "mode_acc": mode_acc,
    "wass_point": wass_point,
}
# Three classes, not two, and the boundary was set by the numerical check in
# test_scoring.py rather than by inspection -- it caught ev_rmse and mode_acc
# being filed as improper when they are proper for a functional.
#
#   STRICT    truth is the UNIQUE minimiser; a point mass can never tie
#   WEAK      proper, but elicits only one functional (the mean, the mode), so
#             any forecast matching that functional ties with the truth --
#             which is how a degenerate predictor draws level with a calibrated
#             one without being penalised
#   IMPROPER  truth is beaten outright; Prop. 1 is the S_0 case
STRICT = ("rps", "logloss", "brier", "spherical")
WEAK = ("ev_rmse", "mode_acc")
IMPROPER = ("s0", "ev_mae", "wass_point")
PROPER = STRICT + WEAK
# What each weakly proper rule actually elicits, used by the strictness test.
ELICITS = {"ev_rmse": "mean", "mode_acc": "mode"}


def score(rule, P, Y):
    """Per-respondent mean over items, NaN answers skipped. Lower is better."""
    s = RULES[rule](P, Y)
    return np.nanmean(np.where(np.isnan(Y), np.nan, s), axis=-1)
