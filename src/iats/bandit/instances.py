"""Causal best-policy-identification instances (the separation testbed).

The minimal mechanism-faithful instance: a Gaussian causal bandit with ``m`` relevant
(reward-ancestor) interventions and ``k`` branch (reward-non-ancestor) interventions.
Intervening on a non-ancestor leaves the reward at its observational baseline, so the
graph-aware learner identifies those for free and excludes them; the blind learner must
rule each out. Returns the mean vectors the two learners face.

The MDP lift replaces arms with policies and means with per-stage Q-bridges; the
relevant/branch split is unchanged, which is why the bandit instance is the right
minimal probe for the leading-order separation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CausalBanditInstance:
    """A causal-bandit BPI instance and the two learners' arm sets.

    Attributes:
        relevant_means: Means of the reward-ancestor interventions (best = max).
        branch_mean: Common reward value of every reward-non-ancestor intervention.
        n_branches: Number of branch interventions the blind learner must rule out.
    """

    relevant_means: list[float]
    branch_mean: float
    n_branches: int

    @property
    def graph_arms(self) -> list[float]:
        """Arm means the graph-aware learner faces.

        Returns:
            The relevant-intervention means (branches excluded).
        """
        return list(self.relevant_means)

    @property
    def blind_arms(self) -> list[float]:
        """Arm means the blind learner faces.

        Returns:
            The relevant means with the ``n_branches`` branch means appended.
        """
        return [*self.relevant_means, *([self.branch_mean] * self.n_branches)]


def chain_with_branches(
    best: float = 1.0,
    competitor: float = 0.5,
    branch_mean: float = 0.2,
    n_branches: int = 0,
) -> CausalBanditInstance:
    """Standard separating instance: one competitor plus ``n_branches`` distractors.

    Args:
        best: Mean of the optimal (reward-ancestor) intervention.
        competitor: Mean of the runner-up reward-ancestor intervention.
        branch_mean: Baseline reward of each reward-non-ancestor branch (< best).
        n_branches: Number of branch interventions.

    Returns:
        The :class:`CausalBanditInstance`.
    """
    return CausalBanditInstance([best, competitor], branch_mean, n_branches)
