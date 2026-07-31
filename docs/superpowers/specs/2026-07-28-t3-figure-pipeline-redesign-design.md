# T3 Figure-Extraction Pipeline Redesign

Date: 2026-07-28
Status: Approved, not yet implemented
Scope: `figure_refine_pipeline/` (extends in place — Approach A)

## Problem

The current `figure_refine_pipeline/` extends the frozen Pass-1 extraction
(`prompts/prompty_frozen.md`) by re-reading Priority-3 (figure-derived)
fields from a high-resolution crop of their source figure. Investigation
this session traced three compounding problems that this redesign
addresses:

1. **Pass-1's own `[Priority 3]` tier tagging is unreliable run-to-run.**
   Confirmed on Paper12: one Pass-1 run calculated `Longitudinal Stress at
   HEL` via the documented Priority-2 formula
   `σ_HEL = 0.5 × ρ₀ × c_l × u_HEL` (baseline evidence:
   `artifacts/gemini_file_api_30/development/Paper12/evidence_source.csv`,
   `CALCULATED (P2)`, giving `3.349 GPa` for VA0). Another Pass-1 run on
   the same paper could not recover `u_HEL` for the VA0 sample group from
   text that draw, and instead fell back to visually reading Fig. 5(a),
   landing on `~1.0 GPa` (see
   `figure_refine_pipeline/artifacts/t3_batch/development/Paper12/raw_response_pass1.md`).
   The field's tier is a function of what Pass-1 happened to find that
   run, not a stable property of the field.

2. **LLM arithmetic drifts even when the correct Priority-2 strategy is
   used.** Repeated draws of the same formula with the same inputs
   produced `3.348`, `3.349`, `3.350`, `3.351` — small, avoidable noise
   from the LLM re-deriving the multiplication each call rather than a
   deterministic computation.

3. **The crop-based refine step behaves correctly but has no fallback.**
   `refine_field()`'s targeted prompt correctly declines to invent values
   when a figure doesn't contain the needed point — confirmed directly
   from
   `figure_refine_pipeline/artifacts/t3_batch/development/Paper12/raw_response_refine_Longitudinal.md`,
   where the model reported "no σ_HEL plotted for 0% pre-strain" instead
   of guessing. But `apply_patch()` unconditionally overwrites the
   field with whatever comes back — including blanks — with no path back
   to the Priority-2 calculation that was actually correct, and it
   destroys the original Pass-1 evidence text in the process (overwrites
   `Notes` with a fixed `"[Priority 3 - HIGH-RES REVIEW]..."` string).

Separately, a standalone CV script (`digitize_figure.py`: flood-fill
connected-component marker detection + axis calibration) combined with an
LLM-matching step (`hybrid_match.py`: CV measures pixel positions
deterministically, LLM only assigns detected markers to sample/shot IDs)
achieved 95.0% graded accuracy with perfect 5/5 run-to-run reproducibility
on Paper1 Figure 4 — the most reproducible result of the investigation —
but was never integrated into the pipeline, and was validated on exactly
one figure (a discrete-marker scatter plot). It must not be assumed to
generalize to other chart types without an explicit routing/fallback
layer, since the known paper set is not assumed to represent all figure
types the pipeline will ever see.

## Constraints

- The frozen Pass-1 prompt (`prompts/prompty_frozen.md`) is byte-identical
  and out of scope. It remains the sole source of truth for T1 (direct
  text) fields and stays the first attempt for T2 (calculated) fields.
- Build strategy is **Approach A: extend in place.** The existing,
  already-validated parts of `figure_refine_pipeline/` — figure
  localization (`find_figure_page`, `locate_figure_bbox`), crop rendering
  (`render_figure_crop`), and the `refine_field()` prompt itself — are not
  rewritten. New logic is added as new modules that `run_pipeline()` calls
  into, plus a targeted fix to `apply_patch()`.
- Rollout is paper-by-paper against the 30-paper corpus, starting with a
  single-field pilot on Paper12, not a batch run across all papers at
  once — consistent with the billing sensitivity raised earlier in this
  investigation.

## Architecture

```
Pass 1 (unchanged, frozen prompt)
   │
   ▼
Field triage — for each non-T1 field, does the evidence text show
BOTH a calculation source AND a figure reference? (detected from the
evidence text pattern itself, e.g. "Page 6, Fig. 5(a) & CALCULATED" —
not a hardcoded per-paper list, so it generalizes to unseen papers)
   │
   ├─ No dual-path → leave as-is (today's T1/T2 behavior, untouched)
   │
   └─ Yes, dual-path → T2 GATE (narrow scope, see below)
         │
         ├─ Formula + input fields identified, all inputs resolvable
         │  from extracted T1 data → recompute deterministically in
         │  Python (never LLM arithmetic) → done, figure never touched
         │
         └─ Inputs not resolvable for some/all samples → those
            samples proceed to T3, per-sample (not necessarily the
            whole field):
               figure localization (existing, unchanged)
                 → figure-type classification (new LLM call)
                     → discrete-marker → CV detector (validated module)
                     → anything else  → CV flagged unavailable
                 → LLM crop read (existing refine_field, unchanged
                   prompt) — always runs regardless of CV availability;
                   matches against CV-detected markers when available,
                   measures and matches itself when not
                 → repeat: 3x if a CV anchor exists, 5x if not
                 → reconcile (tolerance-cluster + median-of-majority)
                 → patch (fixed apply_patch, see "Patch behavior" below)
```

### T2 gate (narrow scope)

Triggers only on fields where Pass-1's own evidence text shows a
dual-path source (both a calculation and a figure reference) — this is
the "narrow" scope agreed for v1, as opposed to a "broad" gate that would
re-verify every Priority-2 field regardless of whether it ever risks
falling back to a figure. **The broad gate is an explicit possible future
change**, to be revisited only if paper-by-paper testing shows arithmetic
drift is a problem beyond the dual-path fields we've directly observed —
not built in v1.

The gate itself splits the task the same way the CV/LLM split works for
T3: the LLM identifies *which* equation and *which* named input fields
apply (a semantic task it's suited for); code then computes the number
deterministically from those inputs once pulled from the already-extracted
T1 table (removing LLM arithmetic as a noise source). This resolves
per-sample, not per-field: on Paper12, VA0.6/VA5.5 would resolve through
the T2 gate (their `u_HEL` is explicitly stated in text), while VA0 alone
falls through to the T3 path — matching what was actually observed.

### Figure-type classification + CV routing

A new LLM call classifies the located figure crop as one of:
discrete-marker / line-with-markers / continuous-curve / bar / box-plot /
non-chart. Only discrete-marker has a real CV detector in v1 (the module
validated on Paper1 Fig. 4). For every other type, the routing layer
reports "CV not available for this chart type" and the pipeline proceeds
LLM-only — the routing logic exists so future CV modules can be added
without re-plumbing, but no other detector is built in v1.

### Repeated sampling + reconciliation

3 draws when a CV anchor exists (LLM is only doing shot-matching against
a fixed CV measurement), 5 draws when it doesn't (LLM must both measure
and match, needs more samples to average out noise).

Reconciliation uses tolerance-based clustering, not majority-vote-on-exact-
value — continuous physical quantities essentially never repeat exactly
(observed spread: `1.040, 1.042, 1.043` across draws of the same true
value). Default clustering tolerance: **2% relative** (tunable; will be
revisited once more papers produce real spread data). The reported value
is the median of the largest cluster. If the largest cluster covers less
than **60%** of the draws, the field is flagged low-confidence rather than
resolved — a near-even split signals a systematic strategy disagreement
(as seen in the Paper12 P2/P3 switching bug), which repeated sampling
alone cannot fix.

Outlier draws (e.g. a single `1.02` among four `~3.50` draws) are excluded
from the reported value but never silently discarded — their raw evidence
text is logged alongside the field, since an excluded outlier's evidence
text is exactly what previously revealed the P2/P3 strategy-switching bug
on Paper12; discarding it would have hidden that finding.

### Patch behavior (`apply_patch()` fix)

Current behavior unconditionally overwrites the target field, including
with blanks, and destroys the original Pass-1 `Notes` text. New behavior:

- Only overwrite the field if reconciliation produced a usable
  (non-low-confidence) value.
- If low-confidence, leave Pass-1's original value in place — do not
  blank it — and set a separate "needs review" flag instead.
- Evidence text is appended to, not replaced, so the original Pass-1
  reasoning survives alongside the patch's.

### Audit trail

A new per-paper JSON log recording, per field/sample: which path was used
(T1 / T2-direct / T2-gated-calc / T3-CV-anchor / T3-LLM-only), every raw
draw and its evidence text, which draws were excluded as outliers and why,
and the final confidence level.

## Error handling

- T2 gate can't identify a formula → treated as genuine T3, proceeds to
  the figure path (today's existing fallback, unchanged).
- Figure localization failure (existing `RuntimeError` paths in
  `locate_figure_bbox`) → skip the patch, keep Pass-1's value, as today.
- Figure-type classification returns non-chart → skip CV, go straight to
  the LLM-only 5-draw path.
- CV detector failure (e.g. no markers found above threshold) → treated as
  "CV not available" for that field, falls back to the LLM-only 5-draw
  path, failure reason logged.
- Reconciliation below the 60% majority threshold → leave Pass-1's
  original value untouched, flag "needs review," do not patch.

## Testing / rollout plan

1. Pilot on Paper12, single field (`Longitudinal Stress at HEL`) to
   validate the T2 gate mechanism end to end against the known failure
   mode.
2. Full Paper12 run across all T3-candidate fields. Each field's figure
   goes through the same figure-type classification step (§3) — no field
   is assumed ahead of time to be CV-eligible, including fields like
   `Spall Strength` where a compatible discrete-marker figure has been
   validated on a different paper (Paper1 Fig. 4).
3. Expand paper-by-paper across the 30-paper corpus, refining the
   figure-type classifier and adding CV modules only as new chart types
   are actually encountered — not built ahead of need.
4. Compare each paper's result against the existing AI-graded baseline
   (`results/gemini_file_api_30/development_numeric_vertex_ai_all/ai_numeric_field_scores.csv`)
   as it goes.

## Explicitly deferred (not in v1)

- Broad T2 gate (verifying every Priority-2 field, not just dual-path
  ones) — possible future change, revisit if paper-by-paper testing shows
  need.
- CV detector modules for line/bar/box-plot chart types.
- Running the full 30-paper batch at once.
- Any change to the frozen Pass-1 prompt.

## Literature grounding

- Self-Ensembling Vision-Language Models for Chart Data Extraction
  (arXiv 2605.27298) — per-cell median aggregation over repeated VLM
  samples, MAD-based confidence; directly informs the reconciliation
  design above.
- Scatteract (arXiv 1704.06687) — detector-based marker finding for
  scatter plots; the standard reference for the CV detector approach.
- Information Extraction from Diverse Charts in Materials Science /
  RCLS metric (OpenReview vj8dqNrzEe) — documents how varied and
  difficult real materials-science figure types are; supports treating
  figure-type generalization as a routing problem rather than assuming
  one detector covers everything.
- ComProScanner (arXiv 2606.00065) — VLM-based figure-extraction agent in
  a similar domain, using figure classification before dispatch.
