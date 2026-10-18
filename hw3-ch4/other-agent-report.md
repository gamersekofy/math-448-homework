I checked the parallel report against my own re-run and the PDF. It is **mostly right**, but it has two factual errors and misses several fixes. All numerical results still reproduce exactly — no computed answer changes.

## What the parallel report got right

- **Q3 sign error** in the constant \(C\): `+ log(...)` should be `- log(...)`.
- **Q13(g) baseline**: 55.6% is the full-data baseline; on the 2009–2010 test set, always-Up is **58.65% (61/104)**.
- **Q13(j) overgeneralization**: LDA on `Lag2+Lag2²` is 61.54%, not 62.5%.
- **Q15(c) snippet mismatch**: the PDF shows `Power2(10,3)`, etc., but the script only calls `Power2(3,8)` and computes part (c) with `Power3`.
- **“0.15 shares”** should be “0.15 billion shares.”
- **Title** “Chapter 4: Statistical Learning” is optional; the chapter is “Classification.”

## Where the parallel report is wrong

**1. “The printed C gives complex roots” is false.**  
With the printed plus sign, the polynomial is \(0.375x^2+1.875x+1.5369\). Its discriminant is **+1.2103**, so the roots are **real**: \(-3.9668\) and \(-1.0332\). The sign error is still real; only this consequence is wrong.

**2. “The 315 fits, the rank table, and the K=47 sweep are all consistent” is false.**  
The table reports Lag2 KNN as **0.596 at K=20**, while the sweep finds **K=47 at 0.615**. If `47` is added to `KNNS`, **three** KNN entries change:

- `Lag2`: 0.596 (K=20) → **0.615 (K=47)**
- `Lag2+Lag2²`: 0.596 (K=13) → **0.606 (K=47)**
- `Lag1+Lag2+Volume`: 0.596 (K=3) → **0.606 (K=47)**

The fit count becomes **330**, not 315. Alternatively, keep the 17-value grid and explicitly say the table uses that grid while Figure 5 is a separate finer 1–50 sweep.

**3. Its corrected 62.5% list is incomplete.**  
The complete set of 62.5% entries is:

- Logit on `Lag2`
- LDA on `Lag2`
- Logit on `Lag2+Lag3`
- LDA on `Lag2+Lag3`
- Logit on `Lag2+Lag2²`
- QDA on `Lag2+Lag2²`
- KNN on `Lag1+Lag2` at `K=25`

The report’s “only Logit/LDA on Lag2 and QDA on Lag2+Lag2²” omits the `Lag2+Lag3` and KNN entries.

## What the parallel report missed

- **Q3 — “\(\delta_k\) linear in \(x\)” is false.** When variances are equal, \(\delta_k\) still has the \(-x^2/(2\sigma^2)\) term; it is common to all classes and cancels from \(\delta_k-\delta_l\), so the **boundary** is linear, not \(\delta_k\) itself.
- **Q3 Figure 1 caption:** \(\delta_1-\delta_2 = -(0.375x^2+1.875x+0.1506)\), not \(+\). The roots are the same, but the equality is written backwards.
- **Q3 “solutions.py verifies numerically…”** The script computes the roots but does not print or assert the residual. Add the check or reword.
- **Q13 QDA/NB explanation:** they predict Up everywhere because the fitted quadratic has **no real roots** (negative discriminant \(\approx -3.1\times10^{-3}\); complex roots \(\approx -2.71\pm2.70i\)), not merely because the variances are “close enough.”
- **Q13 volume wording:** “plateau above 1–2” is misleading; yearly mean Volume rises from 1.0 (2000) to 1.9 (2005) to 5.0 (2008) and 5.6 (2009), then falls to 4.6 (2010). It is roughly exponential growth to 2008–2009.
- **Minor wording:** “priors near 0.5 the two posteriors coincide” and “fitted probabilities hover so close to 0.5…” are imprecise.

# Final fix list

## Must fix

1. **Problem 3, page 2 — \(C\) sign.**  
   Change `+ log(πkσl/(πlσk))` to **`− log(πkσl/(πlσk))`**.  
   (Or rewrite the whole equation as the un-negated \(\delta_k-\delta_l=0\).)

2. **Problem 3, page 2 — “\(\delta_k\) linear” sentence.**  
   Replace “Only when \(\sigma_k^2\) does not depend on \(k\) is \(\delta_k\) linear in \(x\)” with:  
   “Only when \(\sigma_k^2\) does not depend on \(k\) does the \(x^2\) term become common to all classes and cancel from \(\delta_k-\delta_l\), so the decision boundary is linear; \(\delta_k\) itself remains quadratic.”

3. **Problem 13(g), page 8 — test-set baseline.**  
   Replace “worse than the 55.6% ‘always Up’ baseline (50.0%) as well” with:  
   “well below the 58.7% always-Up baseline on these 104 weeks (61/104), and essentially chance.”

## Should fix

4. **Problem 13(j), pages 9–10 — \(K=47\) inconsistency.**  
   Preferred: add `47` to `KNNS`, rerun, and update the three KNN entries above; change “315 fits” → **“330 fits”** and “315 models” → **“330 models.”** The overall best remains QDA at 0.644.  
   Minimal alternative: keep 315 and state that the table uses the 17-value grid while Figure 5 is a separate finer 1–50 sweep.

5. **Problem 13(j), page 9 — 62.5% list.**  
   Replace the current “logistic or LDA on Lag2 or Lag2+Lag2²” phrasing with the complete list above, noting that LDA on `Lag2+Lag2²` is 61.5%.

6. **Problem 13(f)/(h), pages 8–9 — QDA/NB explanation.**  
   Replace “variances/priors are close enough to equal” with the no-real-roots explanation (negative discriminant; complex roots \(\approx -2.71\pm2.70i\)).

7. **Problem 15(c), page 11 — `Power2` calls.**  
   Add to `solutions.py`:
   ```python
   print("(c) Power2 outputs:")
   Power2(10, 3)
   Power2(8, 17)
   Power2(131, 3)
   ```
   The stored `Power3` values can remain.

## Polish

8. Figure 1 caption: change “the roots of \(\delta_1-\delta_2=0.375x^2+1.875x+0.1506=0\)” to “the roots of \(0.375x^2+1.875x+0.1506=0\) (equivalently, \(\delta_1-\delta_2=0\)).”
9. Page 7: “0.15 shares” → “0.15 billion shares”; “plateau above 1–2” → “rising to about 1.0 by 2000 and 5.0–5.6 by 2008–2009.”
10. Page 3: “solutions.py verifies numerically…” → add an assert/print, or say “substituting the roots gives residuals ≤ \(2\times10^{-15}\).”
11. Page 8: “priors near 0.5 the two posteriors coincide” → “with one predictor both reduce to a single threshold, and here they select the same one.”
12. Page 8: “fitted probabilities hover so close to 0.5…” → “most fitted probabilities sit just above 0.5, so the 0.5 rule predicts Up for 987 of 1089 weeks.”
13. Optional title change: “Chapter 4: Statistical Learning” → “Chapter 4: Classification” (keep “MATH 448: Statistical Learning” in the subtitle).

The full report is saved at `./FINAL_VERIFICATION.md`. One caveat: I could not verify the parallel report’s `.typ` line numbers because the `.typ` source wasn’t provided to me; I verified the corresponding content in the PDF and the code. Once your coding agent applies these, send the revised PDF/`.typ` and I’ll re-check it.
