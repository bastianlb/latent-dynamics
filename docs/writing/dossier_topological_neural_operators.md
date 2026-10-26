# Style dossier: Topological Neural Operators
Bastian, Leventhal, Hajij, Birdal. arXiv 2606.09806 (cs.LG, June 2026). Source: arxiv HTML. Related Work is in Appendix H.

## Abstract
- Method-first (contrast with the problem-first hybrid_fmaps opener). Sentence 1 is the contribution:
  > "We introduce Topological Neural Operators (TNOs), a principled framework for operator learning on
  > cell complexes that lifts neural operators (NOs) from functions on points and/or edges to
  > topological domains."
- 6 sentences, ~162 words, **zero numerals**. Arc: introduce -> how (DEC, cross-dimensional coupling) ->
  design principle ("decouple where information flows ... from how it is transformed (which is
  learned)") -> "We further propose Hierarchical TNOs (HTNOs)" -> unification claim ("recovers
  standard point- and graph-based NOs as restricted cases") -> qualitative result.
- Ends on a qualitative result, no code line: "Across a range of PDE benchmarks, including
  irregular-geometry flow problems, TNOs and HTNOs improve accuracy; controlled studies further
  isolate the benefits of native higher-rank and topological structure."

## Introduction
- Hook (generic importance, then named applications):
  > "Modeling and predicting complex physical systems is a central challenge in science and
  > engineering. Many systems, from fluid dynamics to electromagnetism and elasticity, are governed by
  > partial differential equations (PDEs), whose solutions ... underpin applications such as digital
  > twins, design optimization, and real-time simulation [8]."
- Funnel in 3 paragraphs: classical solvers (accurate, costly) -> "Operator learning addresses this by
  ..." + NOs/FNOs -> gap paragraph opened by "Despite these successes, existing NOs are
  fundamentally point-centric: ..." with a concrete list ("Potentials live on vertices, circulations
  on edges, fluxes on faces, and densities in volumes"). Fix is signalled by "As a remedy, we
  introduce ...". Then a design-principle sentence ("Our key design principle is to separate where
  information flows ... from how it is transformed") and an "(i) ... (ii) ... (iii)" benefit list.
- Figure 1 sits in the Introduction after the remedy paragraph (not confirmed page 1). A four-panel
  schematic labelled (i)-(iv): complex, multi-rank signals, the layer with its four learned blocks,
  predicted field. Caption is a single descriptive sentence chain, no result claims: "A TNO layer (iii)
  couples ranks through fixed DEC operators ... and learns the rank-wise channel mixing in four
  blocks, signifying gradient, curl, harmonic, and self maps ..."
- Contributions: a motivating sentence ("This motivates a minimal extension of the NO paradigm: keep
  operator learning, but replace ...") then hinge **"In brief, our contributions are:"** and 4 bullets.
  Bullets are **noun phrases, not "We [verb]"**, one sentence each:
  > "A unifying perspective encapsulating existing NOs as special cases of the proposed framework."
  > "Hierarchical HTNOs for efficient multi-scale propagation of global and topological information."
  After the list a one-line summary: "We validate our approach across synthetic and real PDE
  benchmarks." plus a code-release promise.
- A footnote is used for a quick definitional aside (boundary of a boundary is zero).

## Formatting
- Not bold run-in. Structure is numbered sections/subsections; inside them, short unnumbered
  headings ("Remarks", "Architecture", "Setup", "Results") and, in the experiments, per-benchmark
  headings that name the datasets and point to the table: "Poisson-Gauss, Airfoil, Elasticity
  (Tab. 1)". Setup/Results pairs reused verbatim (5.3, 5.4).
- Experiments open with a one-paragraph roadmap, ending with "Datasets descriptions and full training
  protocol are deferred to appendix F." Heavy appendix (A-H) for background, proofs, details.
- Related Work lives in Appendix H, 4 headed paragraphs ("Neural operators", "Neural operators on
  irregular geometries", ...), each a dense citation chain, ending with a pivot paragraph:
  > "What separates these methods from TNO is that physical quantities are still represented as
  > functions on points ...; TNOs explicitly address this gap: ..."
- Numbered formal items (Definition 2.1, (P1)-(P3) properties, Proofs of "[FNO-Recovery]").
- Tables: "Bold: best; underline: second.", dagger for "our runs with official implementations".

## Novelty calibration
- No "to our knowledge", "novel", "unlike". "first" only in "first-class computational entity".
  One "In contrast to the RIGNO ..." in the appendix. Novelty is argued by **unification**
  (existing NOs are special cases, with proofs in the appendix) and by structural claims:
  > "a structure graphs cannot represent and cell complexes encode natively."
  > "By collapsing all quantities to nodes, current neural operators obscure this structure"
- Scope is hedged honestly in results: "TNO is competitive with RIGNO-18 on AF, ... so methods
  saturate near the discretization noise floor". Baselines run with official code are flagged.

## Voice
- Present tense for claims and method; conclusion in past ("We introduced TNOs ..."), then present.
- Active "we" with strong verbs:
  "We introduce", "We further propose", "We validate", "We evaluate".
- Em-dashes are used a bit (~a dozen in main/appendix), mostly as appositive asides; commas, colons
  and semicolons carry most structure. Colon elaborations are frequent ("point-centric: they ...").
- Connectives: Despite, As a remedy, Thus, Rather than, In brief, As a result, This motivates.
- Parallel triplets and lists everywhere ("vertices, edges, faces, and volumes"); (i)/(ii)/(iii).
- Closing: modest vision line ("points toward scientific foundation models ...") and a one-paragraph
  limitations/future-work list; a Broader-impact paragraph that concedes "learned approximations".
