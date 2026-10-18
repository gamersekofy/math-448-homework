// =====================================================================
// MATH 448: Statistical Learning — Homework 3 (Chapter 4: Classification)
// Problems 3, 6, 13, 15
// Compile:  typst compile solutions.typ
// =====================================================================

#let data = json("results.json")
#let fmt(x, d: 3) = {
  if type(x) == "int" {
    return str(x)
  }
  let r = calc.round(x, digits: d)
  if calc.abs(r) < 0.5 * calc.pow(10, -d) { r = 0.0 }
  if calc.abs(r) >= 1000 and r == calc.round(r, digits: 0) { return str(int(r)) }
  str(r)
}
#let pct(x, d: 1) = str(calc.round(x * 100, digits: d)) + "%"
#let pad0(n) = if n <= 0 { "" } else { "0" + pad0(n - 1) }
// Fixed-decimal formatting (Typst's str() drops trailing zeros).
#let fixed(x, d: 3) = {
  if type(x) == "int" { x = float(x) }
  let r = calc.round(x, digits: d)
  if calc.abs(r) < 0.5 * calc.pow(10, -d) { r = 0.0 }
  let neg = r < 0
  let r = calc.abs(r)
  let p = calc.pow(10, d)
  let scaled = calc.round(r * p)
  let ip = calc.floor(scaled / p)
  let fp = calc.round(scaled - ip * p)
  if fp == p { ip = ip + 1; fp = 0 }
  let fs = pad0(d - str(int(fp)).len()) + str(int(fp))
  (if neg { "-" } else { "" }) + str(int(ip)) + (if d > 0 { "." + fs } else { "" })
}
// Compact predictor-set labels for the (j) rank table.
#let shorten(s) = {
  let t = s.replace("Lag1+Lag2+Lag3+Lag4+Lag5", "Lag1–5")
  t = t.replace("_sq", "²")
  t = t.replace("Lag1Lag2", "Lag1·Lag2")
  t = t.replace("logVol", "log(Vol)")
  t = t.replace("+", " + ")
  t
}

#set page(paper: "us-letter", margin: (top: 2.4cm, bottom: 2.2cm, x: 2.4cm), numbering: "1", number-align: center)
#set text(font: "New Computer Modern", size: 10.5pt, lang: "en")
#set par(justify: true, leading: 0.62em)
#show raw: set text(font: "DejaVu Sans Mono", size: 0.86em)
#show heading.where(level: 2): it => pagebreak(weak: true) + it

#let qtext(body) = block(
  fill: rgb(245, 246, 249),
  stroke: (left: 2.4pt + rgb(80, 100, 140)),
  inset: (x: 10pt, y: 8pt),
  radius: (right: 3pt),
  width: 100%, breakable: true, body,
)

#let conf-mat(m) = table(
  columns: 3,
  align: center + horizon,
  stroke: none,
  inset: (x: 11pt, y: 4.5pt),
  table.hline(stroke: 1pt),
  table.header([*Actual ∖ Predicted*], [*Down*], [*Up*]),
  table.hline(stroke: 0.5pt),
  [Down], [#(m.at(0).at(0))], [#(m.at(0).at(1))],
  [Up], [#(m.at(1).at(0))], [#(m.at(1).at(1))],
  table.hline(stroke: 1pt),
)

// ============== title block ==============
#align(center)[
  #v(0.3em)
  #text(size: 17pt, weight: "bold")[Chapter 4: Classification]
  #v(0.35em)
  #text(size: 11pt)[MATH 448: Statistical Learning — Problems 3, 6, 13, 15]
  #v(1.0em)
  #text(size: 12pt)[Uzair Hamed Mohammed]
  #v(0.25em)
  #text(size: 10.5pt)[October 16, 2026]
]
#v(1.4em)

Reproducibility for the applied questions: the script `solutions.py` computes every number and figure shown below.
#block(inset: (x: 12pt, y: 9pt), fill: rgb(247, 247, 249), radius: 3pt)[
```bash
pip install ISLP
python solutions.py          # computes results.json + figs/
typst compile solutions.typ  # produces this PDF
```
]

#outline(title: [Contents], indent: auto)
#pagebreak()

// =====================================================================
// ============== PROBLEM 3 ==============
// =====================================================================
== Problem 3 — Bayes classifier under QDA with p = 1

#qtext[*Question.* This problem relates to the QDA model, in which the observations within each class are drawn from a normal distribution with a class-specific mean vector and a *class-specific covariance matrix*. We consider the simple case where $p = 1$; i.e. there is only one feature. Suppose that we have $K$ classes, and that if an observation belongs to the $k$th class then $X$ comes from a one-dimensional normal distribution, $X ~ N(mu_k, sigma_k^2)$. Prove that in this case, the Bayes classifier is *not* linear. Argue that it is in fact *quadratic*.]

#v(0.2em)
*Setup and discriminant function.* Let $pi_k$ be the prior probability of class $k$. Under $Y = k$ the density of $X$ is the one-dimensional normal density (equation (4.16) with a class-specific variance),
$ f_k (x) = 1/(sqrt(2 pi) sigma_k) exp(-1/(2 sigma_k^2) (x - mu_k)^2), $
and the Bayes classifier assigns $x$ to the class $k$ that maximizes the posterior
$ p_k (x) = Pr(Y = k | X = x) = (pi_k f_k (x)) / sum_(l = 1)^K pi_l f_l (x). $
Taking logarithms and dropping terms that do not depend on the class (the constant $-1\/2 log 2 pi$ appears in every class), we obtain the discriminant function
$ delta_k (x) &= log pi_k - log sigma_k - (x - mu_k)^2/(2 sigma_k^2) \
              &= -x^2/(2 sigma_k^2) + (mu_k/sigma_k^2) x - mu_k^2/(2 sigma_k^2) - log sigma_k + log pi_k. $

Observe that $delta_k (x)$ is a *quadratic function of $x$*: the coefficient of $x^2$, namely $-1\/(2 sigma_k^2)$, depends on the class. Only when $sigma_k^2$ does not depend on $k$ does the $x^2$ term become common to all classes and cancel from $delta_k - delta_l$, so the decision *boundary* is linear; $delta_k$ itself remains quadratic.

*The boundary solves a quadratic equation.* The Bayes decision rule switches from class $k$ to class $l$ exactly where the posteriors tie, i.e. where $delta_k (x) = delta_l (x)$. Writing out the equality,
$ log pi_k - log sigma_k - (x - mu_k)^2/(2 sigma_k^2) = log pi_l - log sigma_l - (x - mu_l)^2/(2 sigma_l^2), $
then expanding the squares and collecting powers of $x$,
$ 0 &= underbrace(1/2 (1/sigma_k^2 - 1/sigma_l^2), A) x^2 + underbrace(mu_l/sigma_l^2 - mu_k/sigma_k^2, B) x \
    & quad + underbrace(mu_k^2/(2 sigma_k^2) - mu_l^2/(2 sigma_l^2) - log((pi_k sigma_l)/(pi_l sigma_k)), C). dot $

This is a *quadratic equation in $x$*, $A x^2 + B x + C = 0$, whose leading coefficient
$ A = 1/2 (1/sigma_k^2 - 1/sigma_l^2) $
vanishes if and only if $sigma_k^2 = sigma_l^2$. Under QDA's class-specific variances, generically $A != 0$, so the equation is genuinely quadratic:

- *Up to two boundary points.* The equation then has (generically) two distinct real roots $x_1 < x_2$, so the decision rule flips classes twice: the class with the *larger* variance dominates at $plus.minus infinity$ (its density spreads out more, so far in either tail $f_k > f_l$ for wide $k$), while the tighter class is selected only on the bounded interval between the two roots. The boundary therefore consists of *two points* rather than a single threshold — a configuration no linear classifier in one dimension could produce. A linear (LDA-style) boundary arises exactly in the degenerate sub-case $A = 0$, i.e. forcing $sigma_k^2 = sigma_l^2$.
- *Degenerate special case.* If the discriminant $B^2 - 4A C <= 0$, the quadratic never changes sign, the tighter-variance class is *never* chosen on a bounded interval, and the classifier is constant — itself a behavior that unequal variances allow but equal variances (with unequal priors aside) do not.

Hence for the QDA model with $p = 1$, log-posterior ratios are quadratic functions of $x$ and the Bayes decision boundaries are the roots of a quadratic: *the Bayes classifier is not linear; it is quadratic.*
#figure(
  image("figs/q3_qda_boundary.png", width: 96%),
  caption: [
    Illustration for $p = 1$, $K = 2$, equal priors $pi_1 = pi_2 = 0.5$, showing the log-posteriors $delta_k (x)$.
    *Left:* equal variances (LDA-like): one linear boundary at $x = 0$.
    *Right:* class-specific variances $sigma_1^2 = 1$, $sigma_2^2 = 4$ (QDA): the boundaries are the two dashed lines $x approx -4.92$ and $x approx -0.08$, the roots of $0.375 x^2 + 1.875 x + 0.1506 = 0$ (equivalently, $delta_1 - delta_2 = 0$). `solutions.py` prints the residual and asserts that $delta_1 - delta_2$ vanishes at these two roots to machine precision.
  ],
) <fig-q3>

// =====================================================================
// ============== PROBLEM 6 ==============
// =====================================================================
== Problem 6 — Logistic regression on study hours and GPA

#qtext[*Question.* We collect data on a statistics class with $X_1$ = hours studied, $X_2$ = undergrad GPA and $Y$ = receives an A. The fitted logistic regression gives $hat(beta)_0 = -6$, $hat(beta)_1 = 0.05$, $hat(beta)_2 = 1$. (a) Estimate the probability that a student who studies for 40 h and has an undergrad GPA of 3.5 gets an A. (b) How many hours would that student need to study to have a 50% chance of getting an A?]

=== (a) Probability of an A

The fitted logit for a student with $x_1 = 40$ hours and $x_2 = 3.5$ GPA is
$ hat(eta) = hat(beta)_0 + hat(beta)_1 x_1 + hat(beta)_2 x_2 = -6 + (0.05)(40) + (1)(3.5) = -6 + 2 + 3.5 = -0.5. $
Applying the logistic function,
$ hat(p) = Pr(Y = "A" | X) = (e^(-0.5)) / (1 + e^(-0.5)) = 1/(1 + e^(0.5)) = 1/(1 + 1.6487) = 0.3775. dot $

The estimated probability that this student receives an A is *$bold(0.3775)$*, i.e. about *37.75%*.

=== (b) Hours needed for a 50% chance of an A

$hat(p) = 0.5$ means even odds, so the fitted logit must vanish:
$ hat(eta) = -6 + 0.05 h + (1)(3.5) = 0 quad ==> quad 0.05 h = 2.5 quad ==> quad h = 50. dot $
So the student would need to study *50 hours*. Check: with $h = 50$, $hat(eta) = -6 + 2.5 + 3.5 = 0$, and indeed $hat(p) = 1/(1 + e^0) = 0.5$ exactly — verified numerically in `solutions.py`.

// =====================================================================
// ============== PROBLEM 13 ==============
// =====================================================================
== Problem 13 — the Weekly data set

#qtext[*Questions.* Using the `Weekly` data set (1,089 weekly percentage returns of the S&P 500, 1990–2010): (a) produce numerical and graphical summaries and comment on any patterns; (b) fit a logistic regression of `Direction` on the five lag variables plus `Volume` using the full data, and identify any statistically significant predictors; (c) compute the confusion matrix and the overall fraction of correct predictions, and interpret the types of mistakes; (d) fit logistic regression on a 1990–2008 training set with `Lag2` as the only predictor, and compute the confusion matrix and fraction of correct predictions on the 2009–2010 held-out data; (e) repeat with LDA; (f) repeat with QDA; (g) repeat with KNN with $K = 1$; (h) repeat with naive Bayes; (i) which method appears best; (j) experiment with different predictor combinations (including transformations and interactions) and with values of $K$ for KNN, and report the best combination found.]

The data is loaded with `ISLP.load_data("Weekly")`: 1,089 observations, 9 variables (`Year`, `Lag1`–`Lag5`, `Volume`, `Today`, `Direction`), with 605 weeks going Up and 484 Down. The majority class ("always predict Up") would be correct 55.6% of the time. All models in (d)–(j) train on 1990–2008 and predict the 104 held-out weeks of 2009–2010.

#let dstats = data.at("q13a_describe")
#let dvars = ("Year", "Lag1", "Lag2", "Lag3", "Lag4", "Lag5", "Volume", "Today")
#let ddig = ("Year": 1, "Lag1": 3, "Lag2": 3, "Lag3": 3, "Lag4": 3, "Lag5": 3, "Volume": 4, "Today": 3)
#let dk = (("Mean", "mean"), ("Std. dev.", "std"), ("Minimum", "min"), ("Maximum", "max"))

=== (a) Numerical and graphical summaries

Summary statistics (lag variables and `Today` in percent, `Volume` in billions of weekly shares):
#table(
  columns: (auto, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
  align: (left, center, center, center, center, center, center, center, center),
  stroke: none,
  inset: (x: 5pt, y: 4pt),
  table.hline(stroke: 1pt),
  table.header([*Statistic*], [Year], [Lag1], [Lag2], [Lag3], [Lag4], [Lag5], [Volume], [Today]),
  table.hline(stroke: 0.5pt),
  ..dk.map(((lab, key)) => ([#lab], ..dvars.map(v => [#fixed(dstats.at(v).at(key), d: ddig.at(v))]))).flatten(),
  table.hline(stroke: 1pt),
) <tbl-q13a-desc>

#figure(
  image("figs/q13a_boxplots.png", width: 96%),
  caption: [Boxplots of `Lag1`–`Lag5` and `Today` split by `Direction` (Down = red, Up = blue). The lag distributions for the two classes overlap almost completely, with only small median shifts; `Today` separates the classes almost perfectly, by construction.],
) <fig-q13-boxplots>
#figure(
  image("figs/q13a_volume.png", width: 88%),
  caption: [Weekly `Volume`, 1990–2010 (light) with the yearly mean (red). Volume trends strongly upward until the 2008–2009 financial crisis and declines afterwards.],
) <fig-q13-volume>
#figure(
  image("figs/q13a_corr.png", width: 74%),
  caption: [Correlations among the numeric variables. The only substantial correlation is `Year`–`Volume` ($r = 0.84$); the lag variables are mutually almost uncorrelated ($|r| <= 0.09$) and only weakly correlated with `Today` ($|r| <= 0.12$).],
) <fig-q13-corr>

*Patterns.*

+ *Volume grows over time:* mean weekly volume rises from roughly 0.15 billion shares (1990) to about 1.0 by 2000 and 5.0–5.6 by 2008–2009, then falls to 4.6 in 2010. This agrees with the strong `Year`–`Volume` correlation of $r = 0.84$, and the growth looks roughly exponential up to 2008–2009.
+ *The lag variables are essentially mutually uncorrelated* (all $|r| <= 0.09$), so multicollinearity among lags is negligible, and the plots show no visible structure among the five lags.
+ *Weak univariate signal:* boxplots of each lag by `Direction` overlap almost fully; e.g. the median of `Lag1` is shifted by only a fraction of a percentage point between Down and Up weeks. Any classifier built only on the lags is therefore expected to hover close to the 55.6% "always Up" baseline.
+ `Direction` is essentially the sign of `Today`, so `Today` separates the classes almost perfectly in the boxplot — but `Today` is not available at prediction time and is excluded from every fitted model.

=== (b) Logistic regression, all six predictors, full data

Fitting a logistic regression (binomial GLM) of `Direction` (Up = 1) on `Lag1`–`Lag5` and `Volume` using all 1,089 weeks:
#let bsm = data.at("q13b_smry")
#let bn = (("const", "Intercept"), ("Lag1", "Lag1"), ("Lag2", "Lag2"), ("Lag3", "Lag3"), ("Lag4", "Lag4"), ("Lag5", "Lag5"), ("Volume", "Volume"))
#table(
  columns: (auto, 1fr, 1fr, 1fr, 1fr),
  align: (left, center, center, center, center),
  stroke: none,
  inset: (x: 6pt, y: 4.2pt),
  table.hline(stroke: 1pt),
  table.header([*Term*], [*Coef.*], [*Std. err.*], [*$z$*], [*$p$-value*]),
  table.hline(stroke: 0.5pt),
  ..bn.map(((key, name)) => (
    [#name],
    [#fmt(bsm.at(key).at("coef"), d: 4)],
    [#fmt(bsm.at(key).at("se"), d: 4)],
    [#fmt(bsm.at(key).at("z"), d: 4)],
    if bsm.at(key).at("p") < 0.05 { strong[#fmt(bsm.at(key).at("p"), d: 4)] } else { [#fmt(bsm.at(key).at("p"), d: 4)] },
  )).flatten(),
  table.hline(stroke: 1pt),
) <tbl-q13b>

*Significant predictors.* At the 5% level, only *`Lag2`* is statistically significant ($hat(beta)_2 = 0.0584$, $p = 0.0296$, positive coefficient: a higher return two weeks ago is associated with a higher probability of an Up week). The intercept is also significant ($p = 0.0019$), mainly reflecting the Up/Down class imbalance rather than any predictor. `Lag1` is borderline ($p = 0.1181$), while `Lag3`, `Lag4`, `Lag5`, and `Volume` are far from significant ($p >= 0.29$).

=== (c) Confusion matrix on the full data

Predicting on the training data (fitted probability > 0.5 → "Up"):
#let cmc = data.at("q13c_confusion")
#grid(
  columns: (auto, 1fr),
  align: (center, left),
  column-gutter: 16pt,
  row-gutter: 8pt,
  conf-mat(cmc.at("matrix")),
  [
    *Training fraction correct:* 611 / 1089 = *#pct(cmc.at("frac"), d: 2)* — barely above the 55.6% "always Up" baseline.

    *Types of mistakes:* the classifier is extremely lopsided. Of its 1089 predictions, only 54 + 48 = 102 are Down; it correctly flags just 54 of the 484 true Down weeks (#pct(cmc.at("down_correct_rate"), d: 1)) while its Up predictions are right on 557 of 605 weeks (#pct(cmc.at("up_correct_rate"), d: 1)). The dominant error is a *false positive for Up*: the market is predicted to go Up when it actually goes Down (430 such weeks), versus only 48 of the opposite type. Most fitted probabilities sit just above 0.5, so the 0.5 rule predicts Up for 987 of the 1089 weeks, and logistic regression here behaves like an "always Up" classifier and misses almost every down week.
  ],
)

=== (d)–(h) Five classifiers on Lag2, 2009–2010 held-out

Each method is trained on 1990–2008 with `Lag2` as the sole predictor; test accuracy is computed on the 2009–2010 weeks. The table columns are (actual Down → predicted Down), (actual Down → predicted Up), (actual Up → predicted Down), and (actual Up → predicted Up):
#let dh = data.at("q13dh_lag2")
#let dho = ("Logit", "LDA", "QDA", "KNN", "NB")
#let dhrows = dho.map(m => {
  let r = dh.at(m)
  let c = r.at("matrix")
  ([#m], [#(c.at(0).at(0))], [#(c.at(0).at(1))], [#(c.at(1).at(0))], [#(c.at(1).at(1))], [#pct(r.at("frac"))])
}).flatten()
#table(
  columns: (auto, auto, auto, auto, auto, auto),
  align: (left, center, center, center, center, center),
  stroke: none,
  inset: (x: 9pt, y: 4.2pt),
  table.hline(stroke: 1pt),
  table.header([*Method*], [*D→D*], [*D→U*], [*U→D*], [*U→U*], [*% correct*]),
  table.hline(stroke: 0.5pt),
  ..dhrows,
  table.hline(stroke: 1pt),
) <tbl-q13dh>

Part (d) — *logistic regression:* 65/104 = *62.5%* correct; it catches 9 of 43 true Down weeks (matrix row 1) and 56 of 61 true Up weeks.
Part (e) — *LDA:* identical to logistic regression here, 65/104 = *62.5%* — with a single predictor both reduce to a single threshold, and on this split they select the same one; the confusion matrix is exactly the same.
Part (f) — *QDA:* 61/104 = *58.65%*; the fitted quadratic `Lag2` boundary has no real roots (discriminant $approx -3.1 times 10^(-3)$; complex roots $approx -2.71 plus.minus 2.70 i$), so it predicts Up for *all 104* test weeks (matrix [[0, 43], [0, 61]]) — i.e. it degenerates to the "always Up" rule.
Part (g) — *KNN, $K = 1$:* 51/104 = *49.04%* — well below the 58.7% always-Up baseline on these 104 weeks (61/104), and essentially chance; with a noisy single predictor and $K = 1$ the decision follows the nearest neighbor's realized direction essentially at random.
Part (h) — *naive Bayes:* 61/104 = *58.65%* — with one predictor it fits essentially the same one-dimensional quadratic boundary as QDA and likewise predicts Up for every held-out week.

=== (i) Best method on the held-out data

Logistic regression and LDA are the best, tying at *62.5%* (65/104): both predict Up 90 of 104 weeks but catch 9 of the 43 true Down weeks. They are identical here because with a single one-dimensional predictor and fairly symmetric classes the linear boundary is the same. QDA and naive Bayes (58.65%) both predict Up for *every* held-out week, and KNN with $K = 1$ is no better than chance (49.04%). All methods are still far from reliable in this noisy financial setting.

=== (j) Predictor combinations, transformations, interactions, and K

I re-fit every method on combinations of predictors and transformations on the same 1990–2008 train / 2009–2010 test split: 15 predictor sets (including `Lag2^2`, `Lag1^2`, the interaction `Lag1*Lag2`, and `log(Volume)`) cross-fitted with the four regression-type methods plus KNN over $K in {1, 2, 3, 5, 7, 9, 10, 11, 12, 13, 14, 15, 17, 20, 25, 30, 47, 50}$ — 330 fits in total. Test accuracies are tabulated below; for KNN, the reported entry is its best accuracy over the values of $K$ tried (the winning $K$ is shown in the last column).
#let rk = data.at("q13j_rank_table")
#let rkrows = rk.enumerate().map(((i, r)) => {
  let f = if i == 0 { r => table.cell(fill: rgb(233, 241, 229))[#r] } else { r => [#r] }
  (
    f([#shorten(r.at("predictors"))]),
    f([#fmt(r.at("Logit"))]),
    f([#fmt(r.at("LDA"))]),
    f([#fmt(r.at("QDA"))]),
    f([#fmt(r.at("NB"))]),
    f([#fmt(r.at("KNN"))]),
    f([#str(r.at("K*"))]),
  )
}).flatten()
#table(
  columns: (2.6fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
  align: (left, center, center, center, center, center, center),
  stroke: none,
  inset: (x: 6pt, y: 4.2pt),
  table.hline(stroke: 1pt),
  table.header([*Predictors*], [*Logit*], [*LDA*], [*QDA*], [*NB*], [*KNN*], [*$bold(K)$*]),
  table.hline(stroke: 0.5pt),
  ..rkrows,
  table.hline(stroke: 1pt),
) <tbl-q13j>

Entries are test-set accuracy (fraction correct on 104 weeks). The overall best is highlighted.
#let cb = data.at("q13j_best")
#grid(
  columns: (auto, 1fr),
  align: (center, left),
  column-gutter: 16pt,
  row-gutter: 8pt,
  conf-mat(cb.at("matrix")),
  [
    *Best combination found:* *#(cb.at("method"))* on `Lag1`, `Lag2`, `Lag1^2`, `Lag2^2` — 67 / 104 = *#pct(cb.at("frac"), d: 2)* correct, the only configuration above 63%. The squared terms let the quadratic class densities model asymmetric tail behavior around zero returns.
  ],
)
#figure(
  image("figs/q13j_knn_sweep.png", width: 86%),
  caption: [`Lag2`-only KNN test accuracy as a function of $K$. Accuracy is erratic ($K = 1$: 49.0%), stabilizes within 55–62% for most larger $K$, and the best single fit is $K = 47$ (61.5%); $K = 20$ gives 59.6%.],
) <fig-q13-knn>

*Interpretation.* Including quadratic terms in `Lag1` and `Lag2` and using QDA pushes the held-out accuracy to 64.4%; several simpler alternatives sit at 62.5% — logistic or LDA on `Lag2`, logistic or LDA on `Lag2`+`Lag3`, logistic or QDA on `Lag2`+`Lag2^2`, and `Lag1`+`Lag2` with KNN at $K = 25$ (LDA on `Lag2`+`Lag2^2` is a notch lower at 61.5%). However, this comparison has strong selection bias: 330 models were fitted and evaluated on the same 104 held-out weeks, so the winner's edge (64.4% vs. 62.5%, i.e. two extra correct weeks) may largely reflect chance, and the single predictor `Lag2` with plain logistic regression or LDA remains the most defensible choice.

// =====================================================================
// ============== PROBLEM 15 ==============
// =====================================================================
== Problem 15 — Power functions and plots

#qtext[*Question.* (a) Write a function `Power()` that prints $2^3$. (b) Create `Power2(x, a)` that prints $x^a$ for any two numbers $x$ and $a$. (c) Using `Power2()`, compute $10^3$, $8^(17)$, and $131^3$. (d) Create `Power3(x, a)` that *returns* $x^a$. (e) Using `Power3()`, plot $f(x) = x^2$ for $x = 1, dots, 10$, labeling the axes and the title, optionally with a log-scale axis. (f) Create `PlotPower(x, a)` that plots $x^a$ against $x$ for a sequence of values of $x$; e.g. `PlotPower(np.arange(1, 11), 3)` plots $1, dots, 10$ against their cubes.]

=== (a) `Power()`

```python
def Power():
    print(2**3)

Power()   # output: 8
```
Calling `Power()` prints `8` (i.e., $2^3 = 8$).

=== (b) `Power2(x, a)`

```python
def Power2(x, a):
    print(x**a)

Power2(3, 8)   # output: 6561
```
Calling `Power2(3, 8)` prints `6561`, the value of $3^8$.

=== (c) Computing powers with `Power2()`

```python
Power2(10, 3)    # output: 1000
Power2(8, 17)    # output: 2251799813685248
Power2(131, 3)   # output: 2248091
```
So $10^3 = bold(1000)$, $8^(17) = bold(2251799813685248)$, and $131^3 = bold(2248091)$ (we also verify these with the returned-value version `Power3` so they can be stored programmatically).

=== (d) `Power3(x, a)` returning a value

```python
def Power3(x, a):
    result = x**a
    return result
```
Unlike `Power2`, this returns the value as a Python object so it can be stored and reused, e.g. `x = Power3(3, 8)` yields `x == 6561`.

=== (e) Plot of $f(x) = x^2$

```python
x10 = np.arange(1, 11)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(x10, Power3(x10, 2), marker="o", lw=1.8)
ax.set_xlabel("x")
ax.set_ylabel("$x^2$")
ax.set_title("$f(x) = x^2$")
fig.tight_layout()
```
I additionally display the same curve on a log-log scale, discussed below.
#figure(
  image("figs/q15_power3.png", width: 92%),
  caption: [(e) Left: $f(x) = x^2$ on a linear scale for $x = 1, dots, 10$. Right: the same data with `ax.set_xscale("log")` and `ax.set_yscale("log")`; the parabola becomes a straight line with slope 2 on the log-log axes.],
) <fig-q15-p3>

=== (f) `PlotPower(x, a)`

```python
def PlotPower(x, a):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(x, Power3(x, a), marker="o", color="#4c72b0", lw=1.8)
    ax.set_xlabel("x")
    if isinstance(a, int):
        ax.set_ylabel(rf"$x^{a}$")
    else:
        ax.set_ylabel(r"$x^a$")
    ax.set_title(rf"$y = x^{a}$")
    fig.tight_layout()
    return fig

PlotPower(np.arange(1, 11), 3)
```
Calling `PlotPower(np.arange(1, 11), 3)` produces #ref(<fig-q15-pp>), with the $x$-axis showing $1, dots, 10$ and the $y$-axis showing $1^3, ..., 10^3$.
#figure(
  image("figs/q15_plotpower.png", width: 74%),
  caption: [(f) `PlotPower(np.arange(1, 11), 3)`: $x$ against $x^3$.],
) <fig-q15-pp>

// =====================================================================
== Appendix — full `solutions.py` listing
// =====================================================================
#text(size: 7.4pt)[#raw(read("solutions.py"), lang: "python")]
