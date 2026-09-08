"""The Frank-Wolfe flow-polytope solver anchors to the closed form, the separation
persists monotonically beyond the chain, the navigating T&S makes no errors on the
tested seeds, the program is concave, and the single-flip solver understates T* on a
misaligned instance.
"""

from __future__ import annotations

import numpy as np

from iats.bandit.chartime import char_time
from iats.mdp.general import (
    MDPState,
    SmallMDP,
    char_time_polytope,
    concavity_violations,
    is_greedy_aligned,
    mdp_track_and_stop,
    optimal_design,
    separation_polytope,
    single_flip_understatement,
)


def _rect(k: int) -> SmallMDP:
    root = MDPState([0.0], [False], [[(1, 0.5), (2, 0.5)]])
    leaf = MDPState(
        [1.0, 0.5] + [0.2] * k, [False, False] + [True] * k, [[] for _ in range(2 + k)]
    )
    return SmallMDP([root, leaf, leaf])


def _nonrect(k: int) -> SmallMDP:
    root = MDPState([1.0, 0.0], [False, False], [[(1, 1.0)], [(2, 1.0)]])
    leaf = MDPState(
        [1.0, 0.5] + [0.2] * k, [False, False] + [True] * k, [[] for _ in range(2 + k)]
    )
    return SmallMDP([root, leaf, leaf])


def _near_value_tie() -> SmallMDP:
    # Local argmax = a0 (1.0>0.5) but Q(a0)=1.0 < Q(a1)=0.5+0.5645: misaligned near tie.
    return SmallMDP(
        [
            MDPState([1.0, 0.5], [False, False], [[(1, 1.0)], [(2, 1.0)]]),
            MDPState([0.0], [False], [[]]),
            MDPState([0.5645], [False], [[]]),
        ]
    )


def test_polytope_anchors_to_closed_form():
    # Two-decision-state chain (both root actions -> same leaf): T* = char_time = 32.
    root = MDPState([1.0, 0.5], [False, False], [[(1, 1.0)], [(1, 1.0)]])
    leaf = MDPState([1.0, 0.5], [False, False], [[], []])
    t = char_time_polytope(SmallMDP([root, leaf]), graph_aware=True)
    assert abs(t - char_time([1.0, 0.5])) < 1.0, t


def test_demo_instances_are_greedy_aligned():
    # The witness instances sit in the aligned regime (solver = best-policy ID there).
    for build in (_rect, _nonrect):
        for k in (0, 6):
            assert is_greedy_aligned(build(k)), (build.__name__, k)
    # A misaligned gated instance is correctly flagged (local != value argmax at root).
    misaligned = SmallMDP(
        [
            MDPState([1.0, 0.0], [False, False], [[(1, 1.0)], [(2, 1.0)]]),
            MDPState([1.5], [False], [[]]),  # L leads here: value(L)=1.0+1.5
            MDPState([9.0], [False], [[]]),  # R leads here: value(R)=0.0+9.0 (better)
        ]
    )
    assert not is_greedy_aligned(misaligned)


def test_separation_persists_and_is_monotone():
    # Rectangular and non-rectangular: exact T* ratio is monotone increasing in k.
    for build in (_rect, _nonrect):
        ratios = [separation_polytope(build(k))["ratio"] for k in (0, 2, 4, 6)]
        assert abs(ratios[0] - 1.0) < 0.02, (build.__name__, ratios)
        # monotone up to the ~1% Frank-Wolfe convergence residual at default iters.
        assert all(x <= y + 0.03 for x, y in zip(ratios, ratios[1:], strict=False)), (
            build.__name__,
            ratios,
        )
        assert ratios[-1] > 1.5, (build.__name__, ratios)


def test_navigating_track_and_stop_makes_no_errors():
    # IA-T&S on the non-rectangular instance identifies every state's best action.
    mdp = _nonrect(4)
    best = [int(np.argmax(s.means)) for s in mdp.states]
    _, pis = optimal_design(mdp, graph_aware=True)
    errors = 0
    for s in range(12):
        chosen, n = mdp_track_and_stop(
            mdp, pis, delta=0.02, rng=np.random.default_rng(s), graph_aware=True
        )
        errors += int(any(chosen[i] != best[i] for i in range(len(mdp.states))))
        assert n < 200_000
    assert errors == 0, errors


def test_objective_is_concave():
    # sup_w min_s g_s is a concave maximisation: no Jensen violations (paper Remark).
    assert concavity_violations(_nonrect(6), graph_aware=False, n_trials=3000) == 0


def test_single_flip_understates_on_misaligned_instance():
    # Near-value-tie, misaligned: single-flip T* << value-coupled T* (Conjecture I).
    tie = _near_value_tie()
    assert not is_greedy_aligned(tie)
    us = single_flip_understatement(tie)
    assert us["understatement"] > 50.0, us
    assert us["t_single_flip"] < us["t_value_coupled"]
