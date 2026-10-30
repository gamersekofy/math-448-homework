# Chapter 5 Homework: Resampling Methods

3, 5, 6, 7

## Conceptual

3. We now review k-fold cross-validation.

  (a) Explain how k-fold cross-validation is implemented.
  (b) What are the advantages and disadvantages of k-fold crossvalidation relative to:
    i. The validation set approach?
    ii. LOOCV?

## Applied

5. In Chapter 4, we used logistic regression to predict the probability of default using income and balance on the Default data set. We will now estimate the test error of this logistic regression model using the validation set approach. Do not forget to set a random seed before beginning your analysis.

  (a) Fit a logistic regression model that uses income and balance to predict default
  (b) Using the validation set approach, estimate the test error of this model. In order to do this, you must perform the following steps: 
    i. Split the sample set into a training set and a validation set.
    ii. Fit a multiple logistic regression model using only the training observations.
    iii. Obtain a prediction of default status for each individual in the validation set by computing the posterior probability of default for that individual, and classifying the individual to the default category if the posterior probability is greater than 0.5.
    iv. Compute the validation set error, which is the fraction of the observations in the validation set that are misclassified. 

  (c) Repeat the process in (b) three times, using three different splits of the observations into a training set and a validation set. Comment on the results obtained. 

  (d) Now consider a logistic regression model that predicts the probability of default using income, balance, and a dummy variable for student. Estimate the test error for this model using the validation set approach. Comment on whether or not including a dummy variable for student leads to a reduction in the test error rate.

6. We continue to consider the use of a logistic regression model to predict the probability of default using income and balance on the Default data set. In particular, we will now compute estimates for the standard errors of the income and balance logistic regression coefficients in two different ways: (1) using the bootstrap, and (2) using the standard formula for computing the standard errors in the sm.GLM() function. Do not forget to set a random seed before beginning your analysis. 

  (a) Using the summarize() and sm.GLM() functions, determine the estimated standard errors for the coefficients associated with income and balance in a multiple logistic regression model that uses both predictors.
  (b) Write a function, boot_fn(), that takes as input the Default data set as well as an index of the observations, and that outputs the coefficient estimates for income and balance in the multiple logistic regression model.
  (c) Following the bootstrap example in the lab, use your boot_fn() function to estimate the standard errors of the logistic regression coefficients for income and balance.
  (d) Comment on the estimated standard errors obtained using the sm.GLM() function and using the bootstrap. 

7. In Sections 5.1.2 and 5.1.3, we saw that the cross_validate() function can be used in order to compute the LOOCV test error estimate. Alternatively, one could compute those quantities using just sm.GLM() and the predict() method of the fitted model within a for loop. You will now take this approach in order to compute the LOOCV error for a simple logistic regression model on the Weekly data set. Recall that in the context of classification problems, the LOOCV error is given in (5.4). 

  (a) Fit a logistic regression model that predicts Direction using Lag1 and Lag2.
  (b) Fit a logistic regression model that predicts Direction using Lag1 and Lag2 using all but the first observation.
  (c) Use the model from (b) to predict the direction of the first observation. You can do this by predicting that the first observation will go up if P(Direction = "Up"|Lag1, Lag2) > 0.5. Was this observation correctly classified?
  (d) Write a for loop from i = 1 to i = n, where n is the number of observations in the data set, that performs each of the following steps: 
    i. Fit a logistic regression model using all but the ith observation to predict Direction using Lag1 and Lag2.
    ii. Compute the posterior probability of the market moving up for the ith observation.
    iii. Use the posterior probability for the ith observation in order to predict whether or not the market moves up.
    iv. Determine whether or not an error was made in predicting the direction for the ith observation. If an error was made, then indicate this as a 1, and otherwise indicate it as a 0

  (e) Take the average of the n numbers obtained in (d)iv in order to obtain the LOOCV estimate for the test error. Comment on the results.
