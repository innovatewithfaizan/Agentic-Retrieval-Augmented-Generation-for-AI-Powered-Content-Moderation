"""Agreement metrics and the evidence grounding check."""

import math

import numpy as np

from subrating import evaluate


def test_perfect_agreement():
    pairs = [("none", "none"), ("mild", "mild"), ("strong", "strong"), ("moderate", "moderate")]
    s = evaluate.summarize(pairs)
    assert s["exact"] == 1 and s["within_one"] == 1
    assert math.isclose(s["kappa_qw"], 1.0)
    assert np.trace(np.array(s["confusion"])) == 4


def test_off_by_one_and_signed_error():
    pairs = [("mild", "moderate"), ("strong", "moderate"), ("none", "strong"), ("mild", "mild")]
    s = evaluate.summarize(pairs)
    assert s["exact"] == 0.25
    assert s["within_one"] == 0.75
    assert math.isclose(s["mean_signed_error"], (1 - 1 + 3 + 0) / 4)
    # rows are labels, columns are predictions
    assert s["confusion"][0][3] == 1


def test_weighted_kappa_matches_hand_computation():
    # 2x2 used inside a 4-level scale: labels none/strong, predictions agree half the time
    m = np.zeros((4, 4), dtype=int)
    m[0, 0], m[0, 3], m[3, 0], m[3, 3] = 1, 1, 1, 1
    # observed weighted disagreement = 2 * 1 / 4 = 0.5; expected = 0.5 -> kappa 0
    assert math.isclose(evaluate.weighted_kappa(m), 0.0, abs_tol=1e-9)


def test_bootstrap_interval_contains_point_estimate():
    per_film = {f"f{i}": [i % 2 == 0, True, False, True] for i in range(30)}
    lo, hi = evaluate.bootstrap_exact(per_film, iters=500)
    point = sum(sum(v) for v in per_film.values()) / 120
    assert lo <= point <= hi


def test_grounding_accepts_real_lines_and_rejects_invented_ones():
    record = {
        "ratings": {
            "violence": {"evidence": ["[00:01:16] Even if you dont stop this bleeding I give you one more round."]},
            "sex": {"evidence": ["This line is not in the film at all, I promise."]},
            "language": {"evidence": []},
            "substances": {"evidence": []},
        }
    }
    found, total = evaluate.grounding(record, "110_97723_133067_million_dollar_baby.xml")
    assert (found, total) == (1, 2)
