# Style dossier: Forecasting Continuous Non-Conservative Dynamical Systems in SO(3)
Bastian*, Rashed*, Navab, Birdal. ICCV 2025 Oral. arXiv 2508.07775.

## Abstract
- Problem-first. One sentence calls the task "fundamental", then pivots on *yet* into an
  enumerated problem list with inline (1)/(2)/(3):
  > "Modeling the rotation of moving objects is a fundamental task in computer vision, yet SO(3)
  > extrapolation still presents numerous challenges: (1) unknown quantities such as the moment of
  > inertia complicate dynamics, (2) the presence of external forces and torques can lead to
  > non-conservative kinematics, and (3) estimating evolving state trajectories under sparse, noisy
  > observations requires robustness."
- 7 sentences, ~180 words, **zero hard numbers**. Claims are capability statements ("agnostic to",
  "robust to", "generalizes well to").
- Arc: problem → "We propose ..." → "Existing ... often rely on ..., limiting ..." → "In contrast,
  our approach ..." → integration claim → result claim → "Code is available here."

## Introduction
- Hook:
  > "Modeling the dynamics of rigid bodies in 3D is fundamental to numerous domains such as
  > tracking, robotic control, trajectory planning, and sensor fusion [...]. While the mathematical
  > foundations of rigid body mechanics are well understood, many modern applications require
  > predicting rotational motion from noisy sensor data when key physical parameters are unknown.
  > This presents significant challenges at the intersection of data-driven and physical modeling."
  Pattern: context with listed domains → "well understood ... but" tension → named intersection.
- Teaser Figure 1 after the second intro paragraph. 3-sentence caption: method name first, then
  failure case, then fix, with "(bottom left)"/"(bottom right)" pointers.
- Funnel: domain → technical crux (with an informal footnote aside: "consider a spinning top
  balancing due to the gyroscopic effect") → contrast with "previous approaches that restrict
  themselves to conservative systems" → remaining tension → "To this end, we contribute the
  following:" + 3 bullets.
  > "We design an integration control signal on the Lie group SO(3) to simultaneously handle
  > denoising input states and serve as a numerical integration path for a CDE; our model learns a
  > solution to the underlying physics based on a path in SO(3)."

## Formatting
- Related Work: one-line roadmap ("We first review the relevant literature on SO(3) filtering,
  representation learning, and PGML.") then bold 2–4 word Title-Case labels ("SO(3) Filtering",
  "Learning Representations in SO(3)", "Physics-Guided Extrapolation"); each paragraph ends with a
  pivot ("However, they do not consider physical laws governing the underlying dynamics.").

## Novelty calibration
- No "to our knowledge"/"first". Contrast with one specific, named prior method, describing what
  it does and exactly where it breaks:
  > "However, this approach has limitations when modeling noisy signals as it is sensitive to the
  > initial momentum estimation and, by design, cannot handle non-conservative systems."
  > "In contrast to previous approaches that restrict themselves to conservative systems, our
  > method can predict rotational object trajectories in various physical scenarios ..."

## Voice
- Present tense, frequent active "we". Medium-long sentences, short ones as pivots. Concessive
  "While X, Y" openers; "In contrast,", "However,", "Instead,", "To this end,", "Unlike [X], [Y]".
  Occasional dash appositive. Contribution bullets close with *why* the design is justified.
