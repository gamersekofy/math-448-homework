#let data = json("results.json")
#let models = data.models
#let rounded(x, d: 1) = str(calc.round(x, digits: d))
#let whole(x) = str(int(calc.round(x)))
#let mse(name) = whole(models.at(name).test_mse)

= Chapter 6 Homework

Linear Model Selection and Regularization

MATH 448

Uzair Hamed Mohammed

Problems 1, 2, and 9

#pagebreak()

= Problem 1

*(a)* Best subset selection has the smallest training RSS, or ties for it.
It checks every set of $k$ predictors, including the sets considered by the
two stepwise methods.

*(b)* There is no guaranteed winner for test RSS. The model with the lowest
training RSS does not necessarily predict new observations best.

*(c)(i) True.* Forward stepwise adds one predictor at a time, so the
$k$-variable model is contained in its $(k + 1)$-variable model.

*(ii) True.* Backward stepwise removes one predictor at a time, so its
$k$-variable model is a subset of its $(k + 1)$-variable model.

*(iii) False.* Backward and forward stepwise can follow different paths;
the backward model need not be contained in the larger forward model.

*(iv) False.* For the same reason, the forward model need not be contained
in the larger backward model.

*(v) False.* Best subset searches separately at each model size, so the
best sets of $k$ and $k + 1$ predictors need not be nested.

= Problem 2

*(a) iii.* Lasso is less flexible than least squares because it constrains
the coefficients. It can improve prediction when its increase in squared
bias is smaller than its decrease in variance.

*(b) iii.* Ridge regression also constrains the coefficients and is less
flexible than least squares. The same bias–variance tradeoff applies.

*(c) ii.* More flexible nonlinear methods generally reduce bias but
increase variance. They improve prediction when the reduction in squared
bias outweighs the increase in variance.

#pagebreak()

= Problem 9

== (a) Training and test sets

I used the College data with Apps as the response and the other 17
variables as predictors. College names were excluded, and Private was
coded as 1 for Yes and 0 for No. A random 70/30 split with seed 0 gave
#(data.split.n_train) training observations and #(data.split.n_test) test
observations.

For tuning, I used the same shuffled 10-fold cross-validation splits
within the training set. Scaling was fitted separately inside each fold,
as was PCA for PCR. The test set was not used to select penalties or
component counts. After tuning, each model was refitted on the full
training set. Test error below means mean squared error (MSE).

== (b) Least squares

The least-squares model using all 17 predictors had test MSE *#mse("Least squares")*.

== (c) Ridge regression

Cross-validation selected $lambda = #(rounded(models.Ridge.lambda, d: 4))$.
The resulting test MSE was *#mse("Ridge")*.

== (d) Lasso

Cross-validation selected $lambda = #(rounded(models.Lasso.lambda, d: 4))$.
The test MSE was *#mse("Lasso")*, with
*#(models.Lasso.nonzero_predictors) nonzero predictor coefficients* out of
17, not counting the intercept.

The reported penalties use scikit-learn's conventions: ridge minimizes
$"RSS" + lambda sum_j beta_j^2$, while lasso minimizes
$frac("RSS", 2 n_"train") + lambda sum_j abs(beta_j)$.
These use standardized predictors and an unpenalized intercept.

== (e) Principal components regression

Cross-validation over $M = 1, dots, 17$ selected
*$M = #(models.PCR.components)$*. The test MSE was *#mse("PCR")*.

== (f) Partial least squares

Cross-validation over $M = 1, dots, 17$ selected
*$M = #(models.PLS.components)$*. The test MSE was *#mse("PLS")*.

== (g) Comparison

#table(
  columns: 3,
  table.header([Method], [Test MSE], [Test RMSE]),
  [Least squares], [#mse("Least squares")], [#rounded(models.at("Least squares").test_rmse)],
  [Ridge], [#mse("Ridge")], [#rounded(models.Ridge.test_rmse)],
  [Lasso], [#mse("Lasso")], [#rounded(models.Lasso.test_rmse)],
  [PCR], [#mse("PCR")], [#rounded(models.PCR.test_rmse)],
  [PLS], [#mse("PLS")], [#rounded(models.PLS.test_rmse)],
)

All five methods explain about 90–91% of the variation in applications in
the test set. Their RMSEs range from about 1288 to 1324 applications, so
prediction is fairly accurate overall, although errors of this size can
still be substantial for smaller colleges.

Least squares and PCR tie for the lowest test error. PCR selected all 17
components, making it equivalent to least squares here. PLS is very close;
ridge and lasso are slightly worse. The largest RMSE is only about 2.8%
higher than the smallest, so the five approaches perform similarly on
this split. Lasso gives a somewhat smaller model, but not better test
predictions. These results do not establish a universal ranking.
