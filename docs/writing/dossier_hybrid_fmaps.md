# Style dossier: Hybrid Functional Maps for Crease-Aware Non-Isometric Shape Matching
Bastian*, Xie*, Navab, Lähner. CVPR 2024. arXiv 2312.03678.

## Abstract
- Problem-first, short punchy opener:
  > "Non-isometric shape correspondence remains a fundamental challenge in computer vision.
  > Traditional methods using Laplace-Beltrami operator (LBO) eigenmodes face limitations in
  > characterizing high-frequency extrinsic shape changes like bending and creases."
- 7 sentences, ~150 words. **Exactly one quantitative sentence**, the final payoff, led by
  *Notably,*: "up to 15% better mean geodesic error ... and up to 45% improvement in scenarios
  with topological noise." Then "Code is available at: [url]".

## Introduction
- Hook:
  > "Establishing dense correspondences between 3D shapes is a cornerstone for numerous computer
  > vision and graphics tasks such as object recognition, character animation, and texture
  > transfer. The complexity of this task varies significantly depending on the nature of the
  > transformation a shape undergoes."
- Problem-taxonomy funnel (rigid → isometric → non-isometric), then the dominant tool, then its
  weakness, then the one prior fix and *its* flaw, then "To address the shortcomings of the bases
  on their own, we propose ...". The contribution is framed as solving the just-described dual
  failure.
- No numbered Figure 1 in the HTML (possibly a conversion artifact). Figure captions are
  comparison-driven and end by naming the setup ("Experiments are conducted with the ULRSSM [12]
  framework on SMAL.").
- `\textbf{Contributions.}` + "Our contributions are as follows:" + 3 bullets:
  > "We introduce a theoretically grounded framework to estimate functional maps between
  > non-orthogonal basis sets using descriptor-based linear systems, a foundational element of
  > nearly all functional map-based learning methods."

## Formatting
- Bold run-in lead-ins throughout (21 instances): "Intrinsic-Extrinsic Methods.", "Functional
  Maps.", "Basis Functions.", "Data Term.", "Regularizer.". Experiments alternate a bold baseline
  paragraph with a bold "Results." paragraph (≥6 times, same label reused).

## Novelty calibration
- Zero occurrences of "to (the best of) our knowledge", "unlike prior", "first to". Contrast is
  carried by plain connectives in factual sentences:
  > "These approaches still use the purely intrinsic LBO eigenfunctions to define the functional
  > maps basis, adding extrinsic information through regularization or additional steps."
  > "...but none of these have been demonstrated suitable for functional correspondence."
- Measured critique with evidence pointer: "our results indicate it does not perform well in
  learned functional map-based pipelines (c.f. Sec. 5.3)."
- Limitations: concede, then justify: "...leads to computational overhead; however, this is
  tractable for the elastic basis size of 60. Performance gains in the expanded form justify this
  trade-off."

## Voice
- Present tense; "we" moderate (≈10 in ~1800 words), always with strong verbs. No em-dashes; a
  spaced en-dash at most twice ("only *intrinsic* properties – those that do not depend on a
  specific embedding of the surface – stay invariant."). Connectives: However, but, while, despite
  this, to this end, for example, notably. Closing line modest ("paving the way for further
  advances ...").
