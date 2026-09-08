"""Intervention-Aware Track-and-Stop (IATS).

Instance-optimal best-policy identification in causal MDPs with a known graph. A known
causal graph lets the learner drop reward-non-ancestor interventions from the
identification problem (do(non-ancestor) = observational baseline, already identified),
shrinking the alternative set and the Garivier-Kaufmann characteristic time. This
package computes those characteristic times, the graph-aware vs structure-blind
separation, and the Track-and-Stop identification algorithm. CPU-only, no GPU, no
training.
"""
