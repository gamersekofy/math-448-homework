#let data = json("results.json")

#let fixed(x, d: 4) = {
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
  let fs = str(int(fp))
  while fs.len() < d { fs = "0" + fs }
  (if neg { "-" } else { "" }) + str(int(ip)) + (if d > 0 { "." + fs } else { "" })
}

#let pct(x, d: 2) = fixed(x * 100, d: d) + "%"

#let sci(x, d: 3) = {
  if x == 0 { return [0] }
  let ax = calc.abs(x)
  let e = 0
  if ax >= 1 {
    while ax >= 10 { ax = ax / 10; e = e + 1 }
  } else {
    while ax < 1 { ax = ax * 10; e = e - 1 }
  }
  let m = if x < 0 { -ax } else { ax }
  [#fixed(m, d: d) #sym.times 10#super[#e]]
}

#let num(x, d: 4) = if x != 0 and calc.abs(x) < 0.001 {
  sci(x, d: d)
} else {
  [#fixed(x, d: d)]
}

#set page(paper: "us-letter", margin: (top: 2.2cm, bottom: 2.0cm, x: 2.3cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10pt, lang: "en")
#set par(justify: true, leading: 0.56em)
#show table: set text(size: 9.5pt)
#show raw: set text(font: "DejaVu Sans Mono", size: 1em)
#show raw.where(block: true): it => block(breakable: false, it)

#align(center)[
  #v(0.2em)
  #text(size: 17pt, weight: "bold")[Chapter 5: Resampling Methods]
  #v(0.3em)
  #text(size: 11pt)[MATH 448 — Problems 3, 5, 6, 7]
  #v(0.8em)
  #text(size: 12pt)[Uzair Hamed Mohammed]
  #v(0.2em)
  #text(size: 10.5pt)[October 29, 2026]
]
#v(1.2em)

// =====================================================================
== Problem 3 — k-fold cross-validation
// =====================================================================

=== (a) Implementation

Randomly divide the $n$ observations into $k$ roughly equal folds
$F_1, dots, F_k$.  For each $j = 1, dots, k$, fit the model on the
observations outside $F_j$, predict the observations in $F_j$, and
record the fraction misclassified.  The $k$-fold estimate is
$
  "CV"_((k)) = 1/k sum_(j=1)^k 1/(|F_j|) sum_(i in F_j) 1(y_i != hat(y)_i^((-j))),
$
where $hat(y)_i^((-j))$ is the prediction for observation $i$ using the
fit that omits fold $j$.

=== (b) Comparisons

*Validation set.*  Advantage: only one fit is required and the split is
simple.  Disadvantage: the estimate has high variance because it depends on
which observations happen to land in the validation set, and training on only
part of the data tends to overestimate the test error.

*LOOCV.*  Advantage: each fit uses $n - 1$ observations and the procedure has
no split randomness, so it has low bias.  Disadvantage: it requires $n$
fits, and the $n$ training sets are almost identical, so the estimate can
have high variance; $k$-fold CV often gives a better bias-variance and
computational compromise.

// =====================================================================
== Problem 5 — validation-set estimate on Default
// =====================================================================

A random seed of 0 was fixed before all splits.  Each validation split uses
5,000 training and 5,000 validation observations.

=== (a) Logistic regression fit

#let q5a = data.at("q5a").at("coef")
#table(
  columns: (auto, 1fr, 1fr, 1fr, 1fr),
  align: (left, center, center, center, center),
  stroke: none,
  inset: (x: 8pt, y: 4.2pt),
  table.hline(stroke: 1pt),
  table.header([*Term*], [*Estimate*], [*Std. error*], [*$z$*], [*$p$-value*]),
  table.hline(stroke: 0.5pt),
  [Intercept], [#num(q5a.at("const").at("coef"), d: 4)], [#num(q5a.at("const").at("se"), d: 4)], [#fixed(q5a.at("const").at("z"), d: 3)], [#sci(q5a.at("const").at("p"), d: 2)],
  [income], [#num(q5a.at("income").at("coef"), d: 4)], [#num(q5a.at("income").at("se"), d: 4)], [#fixed(q5a.at("income").at("z"), d: 3)], [#sci(q5a.at("income").at("p"), d: 2)],
  [balance], [#num(q5a.at("balance").at("coef"), d: 4)], [#num(q5a.at("balance").at("se"), d: 4)], [#fixed(q5a.at("balance").at("z"), d: 3)], [#sci(q5a.at("balance").at("p"), d: 2)],
  table.hline(stroke: 1pt),
)

Both income and balance are statistically significant, and both coefficients
are positive: higher income and higher balance are associated with a higher
probability of default.

=== (b)–(c) Validation-set error

#let q5c = data.at("q5c")
#let q5s = q5c.at("splits")
#let q5b = data.at("q5b")

#block(breakable: false)[
The first split gives
$
  hat("Err")_"val" = 145/5000 = #pct(q5b.at("error"), d: 2).
$
]
The implementation for one split is:
```python
train, valid = train_test_split(np.arange(n), test_size=0.5,
                                random_state=seed)
X = sm.add_constant(Default[["income", "balance"]])
fit = sm.GLM(y.iloc[train], X.iloc[train],
             family=sm.families.Binomial()).fit()
prob = fit.predict(X.iloc[valid])
error = np.mean((prob > 0.5).astype(int) != y.iloc[valid].to_numpy())
```
Repeating the procedure for the three splits:

#table(
  columns: (auto, 1fr, 1fr, 1fr, 1fr),
  align: (left, center, center, center, center),
  stroke: none,
  inset: (x: 8pt, y: 4.2pt),
  table.hline(stroke: 1pt),
  table.header([*Split*], [*Errors*], [*Validation size*], [*Validation error*], [*Accuracy*]),
  table.hline(stroke: 0.5pt),
  ..q5s.enumerate().map(((i, r)) => (
    [Run #(i + 1)],
    [#(r.at("n_errors"))],
    [#(r.at("n_valid"))],
    [#pct(r.at("error"), d: 2)],
    [#pct(1 - r.at("error"), d: 2)],
  )).flatten(),
  table.hline(stroke: 0.5pt),
  [Mean], [], [], [#pct(q5c.at("mean_error"), d: 2)], [#pct(1 - q5c.at("mean_error"), d: 2)],
  table.hline(stroke: 1pt),
)

The validation error ranges from 2.48% to 2.90%, with mean 2.63% and standard
deviation 0.24 percentage points.  The estimate is therefore somewhat sensitive
to the particular training/validation split, which is the main drawback of the
validation-set approach.

=== (d) Adding the student dummy

#let q5d = data.at("q5d")
#let q5ds = q5d.at("splits")
#table(
  columns: (auto, 1fr, 1fr, 1fr, 1fr),
  align: (left, center, center, center, center),
  stroke: none,
  inset: (x: 8pt, y: 4.2pt),
  table.hline(stroke: 1pt),
  table.header([*Model*], [*Run 1*], [*Run 2*], [*Run 3*], [*Mean*]),
  table.hline(stroke: 0.5pt),
  [income + balance],
  ..q5s.map(r => [#pct(r.at("error"), d: 2)]),
  [#pct(q5c.at("mean_error"), d: 2)],
  [income + balance + student],
  ..q5ds.map(r => [#pct(r.at("error"), d: 2)]),
  [#pct(q5d.at("mean_error"), d: 2)],
  table.hline(stroke: 1pt),
)

Including the student dummy does not reduce the validation error: it is higher
in all three splits, by 0.02 to 0.12 percentage points (mean increase 0.07
percentage points).  The student coefficient is statistically significant and
negative after adjusting for income and balance, but this does not translate
into better out-of-sample classification.

// =====================================================================
== Problem 6 — bootstrap standard errors
// =====================================================================

=== (a) Standard errors from the logistic fit

#let q6a = data.at("q6a").at("coef")
#table(
  columns: (auto, 1fr, 1fr),
  align: (left, center, center),
  stroke: none,
  inset: (x: 8pt, y: 4.2pt),
  table.hline(stroke: 1pt),
  table.header([*Predictor*], [*Estimate*], [*Std. error*]),
  table.hline(stroke: 0.5pt),
  [income], [#num(q6a.at("income").at("coef"), d: 4)], [#num(q6a.at("income").at("se"), d: 4)],
  [balance], [#num(q6a.at("balance").at("coef"), d: 4)], [#num(q6a.at("balance").at("se"), d: 4)],
  table.hline(stroke: 1pt),
)

=== (b) Bootstrap function

For a bootstrap index vector $I^* = (i_1, dots, i_n)$, define
$
  "boot_fn" (I^*) = (hat(beta)_"income"^(*), hat(beta)_"balance"^(*)),
$
the two slope estimates from the logistic regression fitted to the resampled
observations $(x_(i_j), y_(i_j))$ for $j = 1, dots, n$:
```python
def boot_fn(data, index):
    sample = data.iloc[index]
    y = (sample["default"] == "Yes").astype(int)
    X = sm.add_constant(sample[["income", "balance"]])
    fit = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    return {"income": float(fit.params["income"]),
            "balance": float(fit.params["balance"])}
```

=== (c) Bootstrap standard errors

#let q6c = data.at("q6c")
Using #(q6c.at("B")) bootstrap samples, the estimated standard errors
are the sample standard deviations of the bootstrap coefficient values:

#table(
  columns: (auto, 1fr, 1fr),
  align: (left, center, center),
  stroke: none,
  inset: (x: 8pt, y: 4.2pt),
  table.hline(stroke: 1pt),
  table.header([*Predictor*], [*Logistic (GLM) SE*], [*Bootstrap SE*]),
  table.hline(stroke: 0.5pt),
  [income], [#num(q6a.at("income").at("se"), d: 4)], [#num(q6c.at("se_income"), d: 4)],
  [balance], [#num(q6a.at("balance").at("se"), d: 4)], [#num(q6c.at("se_balance"), d: 4)],
  table.hline(stroke: 1pt),
)

=== (d) Comparison

The two sets of standard errors agree closely.  For income, the bootstrap
standard error is $4.77 times 10^(-6)$ versus $4.99 times 10^(-6)$ from the
logistic fit; for balance, it is $2.31 times 10^(-4)$ versus
$2.27 times 10^(-4)$.  The bootstrap therefore supports the model-based
standard errors, with only small simulation differences.

// =====================================================================
== Problem 7 — manual LOOCV on Weekly
// =====================================================================

=== (a) Full-data fit

#let q7a = data.at("q7a").at("coef")
#table(
  columns: (auto, 1fr, 1fr, 1fr, 1fr),
  align: (left, center, center, center, center),
  stroke: none,
  inset: (x: 8pt, y: 4.2pt),
  table.hline(stroke: 1pt),
  table.header([*Term*], [*Estimate*], [*Std. error*], [*$z$*], [*$p$-value*]),
  table.hline(stroke: 0.5pt),
  [Intercept], [#fixed(q7a.at("const").at("coef"), d: 4)], [#fixed(q7a.at("const").at("se"), d: 4)], [#fixed(q7a.at("const").at("z"), d: 3)], [#num(q7a.at("const").at("p"), d: 4)],
  [Lag1], [#fixed(q7a.at("Lag1").at("coef"), d: 4)], [#fixed(q7a.at("Lag1").at("se"), d: 4)], [#fixed(q7a.at("Lag1").at("z"), d: 3)], [#num(q7a.at("Lag1").at("p"), d: 4)],
  [Lag2], [#fixed(q7a.at("Lag2").at("coef"), d: 4)], [#fixed(q7a.at("Lag2").at("se"), d: 4)], [#fixed(q7a.at("Lag2").at("z"), d: 3)], [#num(q7a.at("Lag2").at("p"), d: 4)],
  table.hline(stroke: 1pt),
)

At the 5% level, only Lag2 is significant; its positive coefficient indicates
that a higher return two weeks ago is associated with a higher probability of
an Up week.

=== (b)–(c) Fit without the first observation

#let q7b = data.at("q7b").at("coef")
#let q7c = data.at("q7c")
Excluding the first observation gives
$hat(beta)_"Lag1" = #(fixed(q7b.at("Lag1").at("coef"), d: 4))$ and
$hat(beta)_"Lag2" = #(fixed(q7b.at("Lag2").at("coef"), d: 4))$, nearly
identical to the full-data fit.  For the first observation,
$
  hat(P)(Y = "Up" | X_1) = #fixed(q7c.at("p_up"), d: 4) > 0.5,
$
so the model predicts Up, while the observed direction is Down.  The first
observation is therefore misclassified.

=== (d) Leave-one-out loop

For $i = 1, dots, n$, perform the following steps:

+ Fit the logistic model using all observations except $i$.
+ Compute the fitted probability $hat(P)(Y_i = "Up" | X_i)$ from
  that model.
+ Predict $hat(y)_i = "Up"$ if $hat(P)(Y_i = "Up" | X_i) > 0.5$,
  and $hat(y)_i = "Down"$ otherwise.
+ Set $e_i = 1$ if $hat(y)_i != y_i$, and $e_i = 0$ otherwise.

The corresponding loop is:
```python
X = sm.add_constant(Weekly[["Lag1", "Lag2"]])
errors = []
for i in range(n):
    train = np.arange(n) != i
    fit = sm.GLM(y_weekly[train], X[train],
                 family=sm.families.Binomial()).fit()
    p = float(fit.predict(X.iloc[[i]]).iloc[0])
    errors.append(int((p > 0.5) != bool(y_weekly.iloc[i])))
loocv_error = np.mean(errors)
```

=== (e) LOOCV estimate

#let q7e = data.at("q7e")
Averaging the #(q7e.at("n")) indicators gives
$
  hat("Err")_"LOOCV" = #(q7e.at("n_errors"))/#(q7e.at("n")) = #pct(q7e.at("loocv_error"), d: 2).
$
The corresponding accuracy is #pct(q7e.at("loocv_accuracy"), d: 2), which is
slightly below the #pct(q7e.at("majority_baseline"), d: 2) accuracy obtained by
always predicting Up.  Lag1 and Lag2 thus provide essentially no useful
out-of-sample signal for the direction of the market.
