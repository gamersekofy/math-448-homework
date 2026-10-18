# Validation Report — hw3-ch4 (Chapter 4: Classification)

Problems 3, 6, 13, 15.

All numeric results in `results.json` were independently recomputed from the `Weekly`
data set (via `ISLP.load_data`) and reproduced exactly. The issues below concern the
written solution (`solutions.typ`) rather than the computed results.

> **Revision note.** This report was merged with a second, independent verification
> (`other-agent-report.md` / `FINAL_VERIFICATION.md`). Two claims in the first version of
> this report were wrong and are corrected under "Corrections" below. The consolidated
> fixes were then applied on branch `review/hw3-ch4-fixes`; see "Status" at the end.

## Verified correct

- **Q3 proof structure.** The discriminant
  `δ_k(x) = log π_k − log σ_k − (x − μ_k)² / (2σ_k²)` is quadratic in `x`, with a
  class-dependent `x²` coefficient `−1/(2σ_k²)`. The conclusion that the Bayes
  boundary is quadratic (not linear) is correct. Numeric check for the stated example
  (`μ1=−1.5, σ1²=1, μ2=1.5, σ2²=4`) gives roots `[-4.9183, -0.0817]` from
  `0.375x² + 1.875x + 0.1506 = 0` (residual `|δ1−δ2| ≤ 8.9e-16`).
- **Q6.** `η = −6 + 0.05(40) + 1(3.5) = −0.5` → `p = 1/(1+e^0.5) = 0.3775`; `h = 50`.
- **Q13(b).** Coefficients/p-values reproduce (only `Lag2` significant, `p=0.0296`;
  `Lag1` p=0.1181; intercept p=0.0019; `Lag3`–`Lag5`, `Volume` not significant).
- **Q13(c).** Confusion `[[54,430],[48,557]]`, 611/1089 = 56.11%.
- **Q13(d)–(h).** Logit & LDA `[[9,34],[5,56]]` = 62.5%; QDA & NB `[[0,43],[0,61]]`
  = 58.65%; KNN K=1 `[[22,21],[32,29]]` = 49.04%. All reproduce exactly.
- **Q13(j).** Best = QDA on `Lag1+Lag2+Lag1²+Lag2²` = 67/104 = 64.42%.
- **Q15.** Outputs `8`, `6561`, `1000`, `2251799813685248`, `2248091` are correct.

## Corrections to the first version of this report

1. **"The printed `C` gives complex roots" was wrong.** For the *actual* stated example,
   the printed `C` gives `0.375x² + 1.875x + 1.5369`, discriminant `+1.2103`, so the roots
   are **real**: `-3.9668` and `-1.0332` (not `-4.9183, -0.0817`). The `C` sign error is
   still real; only the described symptom was wrong. (The "complex roots" result came from
   a different, hypothetical example.)
2. **"The 315 fits, rank table, and K=47 sweep are all consistent" was wrong.** The main
   grid omits `K=47` (it is only in the separate 1–50 sweep), so the table reported `Lag2`
   KNN as 0.596 at `K=20` while the figure showed `K=47` = 0.615. Resolved by adding `47`
   to the grid (now 330 fits); three KNN rows change (Lag2 → 0.615; `Lag2+Lag2²` → 0.606;
   `Lag1+Lag2+Volume` → 0.606).

## Issues found (consolidated)

1. **Q3 — sign error in the constant term `C` (substantive).** In `solutions.typ:117-118`,
   `C` is written with `+ log((π_k σ_l)/(π_l σ_k))`. Because `A` and `B` are the
   coefficients of the *negated* equation `δ_l − δ_k = 0`, the log term must be
   **subtracted**: `C = μ_k²/(2σ_k²) − μ_l²/(2σ_l²) − log((π_k σ_l)/(π_l σ_k))`.
   The code, `results.json`, and the caption (`… + 0.1506 = 0`) all use the correct sign;
   only the symbolic formula was inconsistent.
2. **Q3 — "`δ_k` linear in `x`" is false.** `solutions.typ:112`. When variances are equal
   the `x²` term is common to all classes and cancels from `δ_k − δ_l`, so the *boundary*
   is linear; `δ_k` itself stays quadratic.
3. **Q3 — Figure 1 caption equality sign.** `solutions.typ:133`: `δ_1 − δ_2 =
   0.375x² + 1.875x + 0.1506` is sign-flipped (`δ_1 − δ_2 = −(0.375x² + 1.875x + 0.1506)`);
   the roots are unchanged.
4. **Q3 — "verifies numerically" overstated.** `solutions.typ:133`. The script computed
   the roots but never checked the residual; a print/assert was added.
5. **Q13(g) baseline wording.** `solutions.typ:275`. 55.6% is the *full-data* baseline; on
   the 2009–2010 test set always-Up is **58.65%** (61/104).
6. **Q13(j) K=47 / fit-count inconsistency.** See Correction 2 above.
7. **Q13(j) 62.5% list overgeneralized.** `solutions.typ:327`: the complete set is
   Logit/LDA on `Lag2`, Logit/LDA on `Lag2+Lag3`, Logit on `Lag2+Lag2²`, QDA on
   `Lag2+Lag2²`, and KNN on `Lag1+Lag2` (K=25); LDA on `Lag2+Lag2²` is 61.5%.
8. **Q13(f)/(h) QDA/NB explanation.** `solutions.typ:274,276`: they predict Up everywhere
   because the fitted 1-D quadratic boundary has **no real roots** (discriminant ≈
   −3.1e-3; complex roots ≈ −2.71 ± 2.70i), not merely because variances are "close enough".
9. **Q15(c) snippet vs. script.** `solutions.typ:358-362` shows `Power2(10,3)` etc., but
   `solutions.py` only called `Power2(3,8)`; the `(c)` calls were added.
10. **Wording/units.** `solutions.typ:203`: "0.15 shares" → "0.15 billion shares"; the
   "plateau above 1–2" description of `Volume` was replaced with the actual yearly means
   (≈1.0 by 2000, 5.0–5.6 by 2008–2009, 4.6 in 2010). `solutions.typ:273`: "priors near
   0.5 the two posteriors coincide" → single-threshold wording. `solutions.typ:246`:
   "probabilities hover so close to 0.5…" → "the 0.5 rule predicts Up for 987 of 1089 weeks".
11. **Title.** `solutions.typ:74`: "Chapter 4: Statistical Learning" → "Chapter 4:
    Classification" (the MATH 448 subtitle is kept).

## Status

All items above were applied on branch `review/hw3-ch4-fixes`; `results.json`, the
figures, and `solutions.pdf` were regenerated. No computed answer changed except the
intended `KNN K=47` addition to the `(j)` grid (three KNN cells, table sort order, and
the fit count 315 → 330); the overall best remains QDA at 64.42%.
