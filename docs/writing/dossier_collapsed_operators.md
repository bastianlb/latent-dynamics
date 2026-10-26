# Style dossier: Collapsed Effective Operators for Higher-order Structures
Krahn*, Bastian*, Garg, Schuller, Birdal. ICML 2026. arXiv 2606.23517.

## Abstract
- Gap-first, then immediately "We introduce":
  > "Higher-order structures are powerful relational modeling tools, yet existing spectral operators
  > decompose the topology into separate ranks, leaving practitioners to fuse the information back
  > to vertices through ad hoc choices. We introduce Collapsed Effective Operators, which condense
  > higher-order degrees of freedom into a single vertex-level operator via Schur complementation of
  > a graded Laplacian."
- 5 sentences, ~120 words, **no hard numbers**. Ends on a capability list ("...improves spectral
  clustering, signal smoothing and enables the inclusion of topological features in neural network
  architectures via positional encoding.").

## Introduction
- Hook grounds the abstraction in a list of domains and unpacks with a colon:
  > "The collective behavior of complex systems, from protein complexes and chemical reactions, to
  > neural circuits and collaborative networks, is fundamentally polyadic: it emerges from the
  > simultaneous interaction of multiple components. While graphs have emerged as a universal
  > language for modeling relational data, they are fundamentally limited through dyadic edges and
  > cannot represent higher-order interactions directly."
- Teaser Figure 1 on page 1 (protein spectral modes, ours vs graph Laplacian). Caption: one bold
  claim sentence, then one dense sentence with concrete specifics (protein ID, residue count,
  citation, what is colored).
- Four-paragraph funnel ending on a precise gap: "None yields a node-level operator that encodes
  the complete influence of the higher-order structure through a single, principled
  construction." → "To address this gap, we propose ..." → "In summary, we contribute:" + bullets.
  > "We show that our devised operator is positive semidefinite and admits a clear spectral
  > interpretation, reducing the effective conductance in nodes that share common cells."

## Formatting
- Terse bold lead-ins, 1–3 words: "Notation.", "Experimental Setup.", "Results",
  "Discussion on Operator Trade-offs."

## Novelty calibration
- No "to our knowledge". Precise gap scoping; concede-then-distinguish:
  > "Extensions such as the Dirac operator, persistent Laplacians, and the multi-order Laplacian
  > address various limitations, yet all share a common shortcoming: they either operate
  > rank-by-rank or aggregate ranks additively..."
  > "We repurpose this tool for a different goal: collapsing topological ranks rather than spatial
  > subsets"
- Defuses an objection by reframing: "However, we demonstrate that this is not a limitation but a
  feature: the effective operator encodes how higher-order modeling assumptions influence
  vertex-level dynamics."

## Voice
- Present tense for own claims; perfect/passive for history. "We" in intro/contributions, third
  person in Related Work. ~20–30 word sentences joined with *yet*, *while*, colons. Dashes rare;
  commas and colons preferred. "This insight/approach/perspective ..." openers.
