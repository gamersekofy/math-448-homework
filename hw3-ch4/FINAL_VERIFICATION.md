# Final Verification Report — MATH 448 HW3

## Scope

I independently re-ran solutions.py end-to-end and reproduced every numeric result from the official ISLP Weekly data. I then checked the parallel agent's report against the PDF and the code.

## Verdict on the parallel report

The parallel report is mostly correct, but it contains two factual errors and misses several fixes. Its main substantive finding — the Problem 3 sign error — is right; its claim that the printed constant gives complex roots is wrong; and its claim that the 315-fit table and the K=47 sweep are "all consistent" is wrong.

### Findings in the parallel report that are correct

1. Q3 sign error (substantive). The printed constant has + log(pi_k sigma_l / (pi_l sigma_k)); because A and B are coefficients of the negated equation, it should be - log(...). The code, results.json, and the caption's 0.1506 all use the correct sign.
2. Q13(g) baseline. 55.6% is the full-data baseline; on the 2009-2010 test set, always-Up is 58.65% (61/104). The dangling "(50.0%)" is confusing.
3. Q13(j) 62.5% overgeneralization. The text's "logistic or LDA on Lag2 or Lag2+Lag2 squared" includes LDA on Lag2+Lag2 squared, which is 61.54%, not 62.5%.
4. Q15(c) snippet mismatch. The PDF shows Power2(10,3), Power2(8,17), Power2(131,3), but solutions.py only calls Power2(3,8); part (c) is computed with Power3.
5. "0.15 shares" units. The table defines Volume in billions of weekly shares, so it should be "0.15 billion shares."
6. Title (optional). "Chapter 4: Statistical Learning" duplicates the course name; the chapter itself is "Classification."

### Errors in the parallel report

1. "The printed C gives complex roots" is false. With the printed plus sign, the polynomial is 0.375x^2 + 1.875x + 1.5369. Its discriminant is +1.2103, so the roots are real: -3.9668 and -1.0332 (not complex). The sign error is still real; only this consequence is stated incorrectly.

2. "The 315 fits, the rank table, and the K-sweep (best K=47) are all consistent" is false. The table reports Lag2 KNN as 0.596 at K=20, while the sweep finds K=47 at 0.615. If 47 is added to KNNS, three KNN entries change:
   - Lag2: 0.596 (K=20) -> 0.615 (K=47)
   - Lag2+Lag2 squared: 0.596 (K=13) -> 0.606 (K=47)
   - Lag1+Lag2+Volume: 0.596 (K=3) -> 0.606 (K=47)
   The fit count becomes 15 x (4 + 18) = 330, not 315. Alternatively, keep the 17-value grid and explicitly say the table uses that grid while Figure 5 is a separate finer 1-50 sweep.

3. The parallel report's corrected 62.5% list is incomplete. It says "only Logit/LDA on Lag2 and QDA on Lag2+Lag2 squared reach 62.5%." The complete set of 62.5% entries is:
   - Logit on Lag2
   - LDA on Lag2
   - Logit on Lag2+Lag3
   - LDA on Lag2+Lag3
   - Logit on Lag2+Lag2 squared
   - QDA on Lag2+Lag2 squared
   - KNN on Lag1+Lag2 at K=25
   LDA on Lag2+Lag2 squared is 61.54%.

### Fixes the parallel report missed

1. Q3: "delta_k linear in x" is false. When variances are equal, delta_k still has the -x^2/(2 sigma^2) term; it is common to all classes and cancels from delta_k - delta_l, so the boundary is linear. The text says delta_k is linear, which is wrong.
2. Q3 Figure 1 caption: delta_1 - delta_2 = 0.375x^2 + 1.875x + 0.1506 has the sign backwards; delta_1 - delta_2 = -(0.375x^2 + 1.875x + 0.1506).
3. Q3 "solutions.py verifies numerically..." The script computes the roots but does not print or assert the residual. Either add the check or reword.
4. Q13 QDA/NB explanation: They predict Up everywhere because the fitted quadratic has no real roots (negative discriminant about -3.1e-3; complex roots about -2.71 +/- 2.70i), not merely because the variances are "close enough."
5. Q13 volume wording: "plateau above 1-2" is misleading; yearly mean Volume rises from 1.0 (2000) to 1.9 (2005) to 5.0 (2008) and 5.6 (2009), then falls to 4.6 (2010). It is roughly exponential growth to 2008-2009.
6. Q13 LDA/logit wording: "priors near 0.5 the two posteriors coincide" is imprecise; with one predictor both reduce to a single threshold, and on this split they select the same one.
7. Q13 "fitted probabilities hover...": the classifier predicts Up for 987 of 1089 weeks; consider saying that directly.

## Final fix list

### Must fix

1. Problem 3, page 2 — C sign.
   Current: + log(pi_k sigma_l / (pi_l sigma_k))
   Replace with: - log(pi_k sigma_l / (pi_l sigma_k))
   (Or rewrite the whole displayed equation as the un-negated delta_k - delta_l = 0.)

2. Problem 3, page 2 — "delta_k linear" sentence.
   Current: "Only when sigma_k^2 does not depend on k is delta_k linear in x."
   Replace with: "Only when sigma_k^2 does not depend on k does the x^2 term become common to all classes and cancel from delta_k - delta_l, so the decision boundary is linear; delta_k itself remains quadratic."

3. Problem 13(g), page 8 — test-set baseline.
   Current: "worse than the 55.6% 'always Up' baseline (50.0%) as well"
   Replace with: "well below the 58.7% always-Up baseline on these 104 weeks (61/104), and essentially chance."

### Should fix

4. Problem 13(j), pages 9-10 — K=47 inconsistency.
   Preferred: add 47 to KNNS in solutions.py; rerun; update the three KNN entries listed above; update "315 fits" to "330 fits" on page 9 and "315 models" to "330 models" on page 10. The overall best remains QDA at 0.644.
   Minimal alternative: keep 315 and change the table note to "best accuracy over the 17-value grid above"; describe Figure 5 as a separate finer 1-50 sweep.

5. Problem 13(j), page 9 — 62.5% list.
   Current: "several simpler alternatives (logistic or LDA on Lag2 or Lag2+Lag2 squared; Lag1+Lag2 with KNN at K=25) sit at 62.5%."
   Replace with: "several simpler alternatives sit at 62.5%: logistic or LDA on Lag2, logistic or LDA on Lag2+Lag3, logistic or QDA on Lag2+Lag2 squared, and KNN on Lag1+Lag2 at K=25. (LDA on Lag2+Lag2 squared is 61.5%.)"

6. Problem 13(f)/(h), pages 8-9 — QDA/NB explanation.
   Replace the "variances/priors close enough to equal" sentence with: "The fitted QDA boundary has no real roots (negative discriminant, about -3.1e-3; complex roots about -2.71 +/- 2.70i), so QDA assigns Up everywhere; on the 104 test weeks that gives [[0,43],[0,61]]. Naive Bayes fits essentially the same one-dimensional quadratic boundary and likewise predicts Up everywhere."

7. Problem 15(c), page 11 — Power2 calls.
   Add to solutions.py, before or after the results block:

       print("(c) Power2 outputs:")
       Power2(10, 3)
       Power2(8, 17)
       Power2(131, 3)

   The stored Power3 values can remain.

### Polish

8. Figure 1 caption: "the roots of delta_1 - delta_2 = 0.375x^2 + 1.875x + 0.1506 = 0" -> "the roots of 0.375x^2 + 1.875x + 0.1506 = 0 (equivalently, delta_1 - delta_2 = 0)."
9. Page 7: "0.15 shares" -> "0.15 billion shares"; "plateau above 1-2" -> "rising to about 1.0 by 2000 and 5.0-5.6 by 2008-2009."
10. Page 3: "solutions.py verifies numerically..." -> add an assert/print in the script, or reword to "substituting the roots gives residuals <= 2e-15."
11. Page 8: "priors near 0.5 the two posteriors coincide" -> "with one predictor both reduce to a single threshold, and here they select the same one."
12. Page 8: "fitted probabilities hover so close to 0.5..." -> "most fitted probabilities sit just above 0.5, so the 0.5 rule predicts Up for 987 of 1089 weeks."
13. Title (optional): "Chapter 4: Statistical Learning" -> "Chapter 4: Classification" (keep "MATH 448: Statistical Learning" in the subtitle).

## Verification evidence

- Q3 roots: -4.918344988, -0.081655012; delta residual <= 2.2e-15.
- Q6: eta = -0.5; p = 0.377540669; h = 50.
- Q13 full logit: coefficients/SE/z/p all match to 4 decimals.
- Q13(c): [[54,430],[48,557]], 611/1089 = 56.1065%.
- Q13(d)-(h): Logit/LDA [[9,34],[5,56]] 65/104; QDA/NB [[0,43],[0,61]] 61/104; KNN [[22,21],[32,29]] 51/104.
- Q13(j): best QDA [[20,23],[14,47]] 67/104; K=47 impacts three KNN rows as listed above.
- Q15: 8, 6561, 1000, 2251799813685248, 2248091.

Note on line numbers: I verified the page-level content in the PDF and the code. I could not verify the parallel report's .typ line numbers because the .typ source was not provided to me; use the page references above.
