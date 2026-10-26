# Writing guide (house style for papers)

Distilled from Lennart Bastian's first-/co-lead-authored papers (2026-10-15):
- *Forecasting Continuous Non-Conservative Dynamical Systems in SO(3)*, ICCV 2025 Oral — arXiv 2508.07775
- *Collapsed Effective Operators for Higher-order Structures*, ICML 2026 — arXiv 2606.23517
- *Hybrid Functional Maps for Crease-Aware Non-Isometric Shape Matching*, CVPR 2024 — arXiv 2312.03678
- *Topological Neural Operators*, arXiv 2606.09806

Per-paper dossiers with verbatim examples: `dossier_so3.md`, `dossier_collapsed_operators.md`,
`dossier_hybrid_fmaps.md`, `dossier_topological_neural_operators.md` in this directory.

## Rules

### Abstract
- **Open on the problem, not on scene-setting.** Sentence 1 states why the domain matters, then
  pivots with *yet* / *while* to the limitation. Sentence 2 (or 3) is "We introduce / We propose X".
  No "Recently, ..." throat-clearing.
- **5–7 sentences, ~120–180 words.**
- **Numbers are rationed.** Zero to one quantitative sentence. If one, it is the single payoff
  line, often introduced with *Notably,*. Never a string of multipliers and correlations.
- Arc: problem → method/study → why prior approaches fall short → "In contrast, our ..." →
  capability/result → (optional) "Code is available at ...".
- **End on a capability or result**, not on a restated contribution or future-work tease.

### Introduction
- **Hook in three sentences.** (1) Broad importance, grounded in a list of 3–4 concrete named
  instances/domains with citations. (2) "While [classical/known fact], [real gap]". (3) Name the
  tension or the intersection where the problem lives.
- **Funnel over 3–5 paragraphs**, each peeling back one layer of difficulty. Critique the closest
  prior work *by name*, at the level of mechanism ("by design, cannot handle ..."), before
  revealing the contribution as the fix: "To address this gap, we ..." / "To this end, ...".
- **Teaser figure on page 1** (Figure 1) is the norm. The caption names the method/idea first,
  then the depicted failure case, then the depicted fix, with spatial pointers ("bottom left").
  A skimming reader should get the contribution from the figure alone.
- **Contributions**: bold run-in `\textbf{Contributions.}` or a hinge sentence ("To this end, we
  contribute the following:" / "In summary, we contribute:"), then **3–4 bullets**, each 1–2
  sentences opening with *We [strong verb]* (introduce, show, demonstrate, design). Each bullet
  says what *and* why it matters (often a semicolon-joined second clause).

### Novelty calibration
- **No "to our knowledge, the first" or superlative priority claims.** None of the exemplars use
  them. Novelty is argued *structurally*: say precisely what prior work does and where it stops
  ("None yields a node-level operator that ..."), then say what we do.
- **Concede, then distinguish.** Credit what the closest work does well before naming the gap.
- Pre-empt an obvious objection by reframing it ("this is not a limitation but a feature: ...")
  only when it is actually true.
- Hedge with specific, evidence-pointing verbs ("our results indicate ... (cf. Sec. 5.3)"), not
  with generic hedges.

### Paragraph formatting
- **Bold run-in lead-ins everywhere** (Related Work, Method, Experiments): 1–3 word Title-Case
  noun phrase, e.g. `\textbf{Functional Maps.}`, `\textbf{Data Term.}`, `\textbf{Results.}`.
  They are signposts, not topic sentences. Reusing a label ("Results.") verbatim is fine.
- Related Work opens with a one-sentence roadmap ("We first review ..."), then labelled
  paragraphs that each end with a one-sentence pivot to what is still missing.
- Experiments: per baseline/setting, a bold-label description paragraph followed by a bold
  `Results.` paragraph.
- Anchor every claim to a figure/section pointer inline ("(see Fig. 3)", "(cf. Sec. 5.3)").

### Voice
- Present tense for our claims; present perfect/past for prior work. Active *we* with strong
  verbs; never "we believe/think". Related Work in third person.
- Medium-to-long, information-dense sentences; short sentences reserved for pivots.
- **Dashes are rare.** Prefer commas and colons for internal breaks; at most an occasional
  spaced en-dash for an appositive aside. Avoid em-dash-heavy prose.
- Workmanlike connectives: *However, while, yet, instead, to this end, notably, in contrast*.
  Avoid *Interestingly, Crucially, It is worth noting that*.
- Colon-introduced elaborations ("is fundamentally polyadic: it emerges from ...").
- Ground abstractions with short lists of concrete named instances.
- Limitations: concede the cost, then say why it is acceptable, in the same sentence pair.
  Conclusion: restate the idea and the settings tested, close with a modest forward-looking line.

## Notes from Topological Neural Operators
- Abstract may open with the contribution itself ("We introduce X, a principled framework for ...")
  rather than a problem sentence, and may carry **zero numbers** (ends on a qualitative result).
  So "zero to one quantitative sentence" holds; a problem-first opener is a default, not a law.
- Contribution bullets can be **noun phrases** ("A unifying perspective ...") introduced by "In brief,
  our contributions are:", instead of "We [verb]" bullets.
- Bold run-in lead-ins are not universal: this paper uses numbered sections plus short plain headings
  (Setup / Results, per-benchmark headings with table pointers) and puts Related Work in an appendix
  with a closing "What separates these methods from X is ..." pivot.
- Novelty by **unification** ("recovers existing methods as special cases", proved in appendix) is a
  further no-superlatives route. Zero "to our knowledge" holds here too.
- Em-dashes appear occasionally as asides; keep them rare.
