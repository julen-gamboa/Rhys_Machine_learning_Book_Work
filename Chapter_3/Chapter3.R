#install.packages("mlr", dependencies = TRUE)
library(mlr)
#install.packages("mlr3", dependencies = TRUE)
library(mlr3)
library(tidyverse)


install.packages("mclust"); data(diabetes, package = "mclust")

diabetesTib = as_tibble(diabetes)

summary(diabetesTib)

ggplot(diabetesTib, aes(glucose, insulin, col = class)) +
  geom_point() +
  theme_bw()

ggplot(diabetesTib, aes(sspg, insulin, col = class)) +
  geom_point() +
  theme_bw()

ggplot(diabetesTib, aes(sspg, glucose, col = class)) +
  geom_point() +
  theme_bw()

#####################
####Exercise 1#######

ggplot(diabetesTib, aes(glucose, insulin, shape = class)) +
  geom_point() +
  theme_bw()

ggplot(diabetesTib, aes(sspg, insulin, shape = class)) +
  geom_point() +
  theme_bw()

ggplot(diabetesTib, aes(sspg, glucose, shape = class)) +
  geom_point() +
  theme_bw()

ggplot(diabetesTib, aes(glucose, insulin, shape = class, col = class)) +
  geom_point() +
  theme_bw()

ggplot(diabetesTib, aes(sspg, insulin, shape = class, col = class)) +
  geom_point() +
  theme_bw()

ggplot(diabetesTib, aes(sspg, glucose, shape = class, col = class)) +
  geom_point() +
  theme_bw()

#####################

#####################
#### Using mlr to train your first kNN model####

#We understand the problem we’re trying to solve (classifying new patients
#into one of three classes), and now we need to train the kNN algorithm to
#build a model that will solve that problem. Building a machine learning
#model with the mlr package has three main stages:
#  1 Define the task. The task consists of the data and what we want to do
#with it. In this case, the data is diabetesTib, and we want to classify
#the data with the class variable as the target variable.
#  2 Define the learner. The learner is simply the name of the algorithm
#we plan to use, along with any additional arguments the algorithm accepts.
#  3 Train the model. This stage is what it sounds like: you pass the task
#to the learner, and the learner generates a model that you can use to make
#future predictions.


###Telling mlr what we’re trying to achieve: Defining the task####

# Begin by defining our task. The components needed to define a task are
# * The data containing the predictor variables
# (variables we hope contain the information needed to make
# predictions/solve our problem)
# * The target variable we want to predict

# For supervised learning, the target variable will be categorical if
# we have a classification problem, and continuous if we have a
# regression problem.

# For unsupervised learning, we omit the target variable from our
# task definition, as we don’t have access to labeled data.

# Our target variable in this problem is to predict the patient class
# using data on glucose, insulin or steady state blood glucose level (sspg)

# The classification model uses the makeClassifTask() function.
# Regression models use the makeRegrTask() function.
# Clustering models use the makeClusterTask() function.

diabetesTask = makeClassifTask(data = diabetesTib, target = "class")

diabetesTask

####Telling mlr which algorithm to use: Defining the learner####

# The first argument to the makeLearner() function is the algorithm that
# we’re going to use to train our model. In this case, we want to use the
# kNN algorithm, so we supply "classif.knn" as the argument.

# The argument par.vals stands for parameter values, which allows us
# to specify the number of k-nearest neighbors we want the algorithm to use.

listLearners()$class

# or list by function

listLearners("classif")$class
listLearners("regr")$class
listLearners("cluster")$class

knn = makeLearner("classif.knn", par.vals = list("k" = 2))

##### Putting it all together: Training the model ######

knnModel = train(knn, diabetesTask)

##### Check how it performs #####

knnPred = predict(knnModel, newdata = diabetesTib)
knnPred

# MMCE is simply the proportion of cases classified as a class other
# than their true class. Accuracy is the opposite of this:
# the proportion of cases that were correctly classified by the model.
# You can see that the two sum to 1.00

performance(knnPred, measures = list(mmce, acc))


#### Holdout cross-validation #####

# If your test set is too small, then the estimate of performance is
# going to have high variance; but if the training set is too small,
# then the estimate of performance is going to have high bias.
# A commonly used split is to use two-thirds of the
# data for training and the remaining one-third as a test set.

holdout = makeResampleDesc(method = "Holdout", split = 2/3,
                            stratify = TRUE)

# The optional argument, stratify = TRUE asks the function to ensure
# that when it splits the data into training and test sets, it tries
# to maintain the proportion of each class of patient in each set.
# This is important in classification problems like ours, where the
# groups are very unbalanced (we have more healthy patients
# than both other groups combined) because, otherwise, we could get
# a test set with very few of one of our smaller classes.

holdoutCV = resample(learner = knn, task = diabetesTask,
                      resampling = holdout, measures = list(mmce, acc))

holdoutCV$aggr

#####################
####Exercise 2#######

holdoutExercise = makeResampleDesc(method = "Holdout", split = 1/10,
                           stratify = FALSE)

holdoutCVExercise = resample(learner = knn, task = diabetesTask,
                     resampling = holdoutExercise, measures = list(mmce, acc))

holdoutCVExercise$aggr

####################

####Calculating a Confusion Matrix#####

calculateConfusionMatrix(holdoutCV$pred, relative = TRUE)

# The absolute confusion matrix is easier to interpret. The rows
# show the true class labels, and the columns show the predicted labels.
# The numbers represent the number of cases in every combination of
# true class and predicted class.
# Correctly classified patients are found on the diagonal of the matrix
# (where true class == predicted class).

# As the performance metrics reported by holdout CV depend so heavily
# on how much of the data we use as the training and test sets, I try
# to avoid it unless my model is very expensive to train, so I generally
# prefer k-fold CV.

####K-fild cross-validation####

# In k-fold CV, we randomly split the data into approximately equal-sized
#chunks called folds. Then we reserve one of the folds as a test set and
# use the remaining data as the training set (just like in holdout).
# We pass the test set through the model and make a record of the relevant
# performance metrics. Now, we use a different fold of the data as
# our test set and do the same thing. We continue until all the folds
# have been used once as the test set. We then get an average of the
# performance metric as an estimate of model performance.

# we can improve this a little by using repeated k-fold CV, where, after the
# previous procedure, we shuffle the data around and perform it again.
# A commonly chosen value of k for k-fold is 10. Again, this depends on
# the size of the data, among other things, but it is a reasonable
# value for many datasets.  If you have the computational power, it is
# usually preferred to use repeated k-fold CV instead of ordinary k-fold.

kFold = makeResampleDesc(method = "RepCV", folds = 10, reps = 50,
                          stratify = TRUE)

kFoldCV = resample(learner = knn, task = diabetesTask,
                    resampling = kFold, measures = list(mmce, acc))

kFoldCV$aggr
kFoldCV$measures.test

####Exercise 3##############
############################

kFold_500 = makeResampleDesc(method = "RepCV", folds = 3, reps = 500,
                             stratify = TRUE)

kFoldCV_500 <- resample(learner = knn, task = diabetesTask,
                       resampling = kFold_500, measures = list(mmce, acc))

kFold_5 <- makeResampleDesc(method = "RepCV", folds = 3, reps = 5,
                           stratify = TRUE)

kFoldCV_5 <- resample(learner = knn, task = diabetesTask,
                     resampling = kFold_5, measures = list(mmce, acc))

kFoldCV_500$aggr

kFoldCV_5$aggr

#############################

####CALCULATING A CONFUSION MATRIX for k-fold CV#####

calculateConfusionMatrix(kFoldCV$pred, relative = TRUE)


####Leave-one-out cross-validation####

# Because the test set is only a single observation, leave-one-out CV tends
# to give quite variable estimates of model performance (because the
# performance estimate of each iteration depends on correctly labeling
# that single test case). But it can give lessvariable estimates of
# model performance than k-fold when your dataset is small.

LOO = makeResampleDesc(method = "LOO")

LOOCV = resample(learner = knn, task = diabetesTask, resampling = LOO,
                  measures = list(mmce, acc))

LOOCV$aggr

calculateConfusionMatrix(LOOCV$pred, relative = TRUE)

#############################
####Exercise 4####


LOO_Strat = makeResampleDesc(method = "LOO", stratify = TRUE)
#Doesn't work

LOO_Reps = makeResampleDesc(method = "LOO", reps = 5)
#Doesn't make use of the reps argument

####################


####Parameters & hyperparameters#####

# hyperparameter: a variable or option that controls how a model makes
# predictions but is not estimated from the data.
# Always use a procedure called hyperparameter tuning to automate
# the selection process (unless it is computationally prohibitive)

# The first thing we need to do is define a range of values over which mlr
# will try, when tuning k:

knnParamSpace = makeParamSet(makeDiscreteParam("k", values = 1:10))

# there are also functions to define continuous and logical hyperparameters

# Next, we define how we want mlr to search the parameter space.

gridSearch = makeTuneControlGrid()

# Next, we define how we’re going to cross-validate the tuning procedure.
# In this case, will use repeated k-fold CV

# The principle here is that for every value
# in the parameter space (integers 1 to 10), we perform repeated k-fold CV.
# For each value of k, we take the average performance measure across all
# those iterations and compare it with the average performance measures
# for all the other values of k we tried. This will hopefully give us the
# value of k that performs best:

cvForTuning = makeResampleDesc("RepCV", folds = 10, reps = 20)

tunedK = tuneParams("classif.knn", task = diabetesTask,
                     resampling = cvForTuning,
                     par.set = knnParamSpace, control = gridSearch)

tunedK$x

# Plot the tuning process
library(ggplot2)

knnTuningData = generateHyperParsEffectData(tunedK)

plotHyperParsEffect(knnTuningData, x = "k", y = "mmce.test.mean",
                    plot.type = "line") +
  theme_bw()

# Now we can train our final model, using our tuned value of k:
tunedKnn = setHyperPars(makeLearner("classif.knn"),
                           par.vals = tunedK$x)

tunedKnnModel = train(tunedKnn, diabetesTask)

#Re-run with new hyperparameter

knn = makeLearner("classif.knn", par.vals = list("k" = 7))

knnModel = train(knn, diabetesTask)

knnPred = predict(knnModel, newdata = diabetesTib)

kFold = makeResampleDesc(method = "RepCV", folds = 10, reps = 50,
                         stratify = TRUE)

kFoldCV = resample(learner = knn, task = diabetesTask,
                   resampling = kFold, measures = list(mmce, acc))

kFoldCV$aggr

calculateConfusionMatrix(kFoldCV$pred, relative = TRUE)

####Including hyperparameter tuning in cross-validation####

# This takes the form of nested CV, where an inner loop cross-validates
# different values of our hyperparameter (just as we did earlier),
# and then the winning hyperparameter value gets passed to an outer CV
# loop. In the outer CV loop, the winning hyperparameters are used for
# each fold.

# Nested CV proceeds like this:

# 1 Split the data into training and test sets (this can be done using the
# holdout, k-fold, or leave-one-out method). This division is called the
# outer loop.

# 2 The training set is used to cross-validate each value of our hyperparameter
# search space (using whatever method we decide). This is called the inner loop.

# 3 The hyperparameter that gives the best cross-validated performance from each
# inner loop is passed to the outer loop.

# 4 A model is trained on each training set of the outer loop, using the best
# hyperparameter from its inner loop. These models are used to make predictions
# on their test sets.

# 5 The average performance metrics of these models across the outer loop are
# then reported as an estimate of how the model will perform on unseen data.

inner = makeResampleDesc("CV")

outer = makeResampleDesc("RepCV", folds = 10, reps = 5)

knnWrapper = makeTuneWrapper("classif.knn", resampling = inner,
                              par.set = knnParamSpace,
                              control = gridSearch)

cvWithTuning = resample(knnWrapper, diabetesTask, resampling = outer)

cvWithTuning

####Using our model to make predictions####
library(tibble)

newDiabetesPatients = tibble(glucose = c(82, 108, 300),
                              insulin = c(361, 288, 1052),
                              sspg = c(200, 186, 135))

newDiabetesPatients

newPatientsPred = predict(tunedKnnModel, newdata = newDiabetesPatients)

getPredictionResponse(newPatientsPred)

# kNN algorithm
# * It makes no assumptions about the data, such as how it’s distributed.
# * It cannot natively handle categorical variables (they must be recoded
# first, or a different distance metric must be used).
# * When the training set is large, it can be computationally expensive to
# compute the distance between new data and all the cases in the training set.
# * The model can’t be interpreted in terms of real-world relationships in
# the data.
# * Prediction accuracy can be strongly impacted by noisy data and outliers.
# * In high-dimensional datasets, kNN tends to perform poorly. In
# brief, in high dimensions the distances between the cases start to
# look the same, so finding the nearest neighbors becomes difficult.

##########################
####Exercise 5, 6, and 7####

data(iris)

irisTask = makeClassifTask(data = iris, target = "Species")

# set the range of k to test
knnParamSpace = makeParamSet(makeDiscreteParam("k", values = 1:15))

# hyperparameter tune control strategy
gridSearch = makeTuneControlGrid()

# CV of tuning procedure
cvForTuning = makeResampleDesc("RepCV", folds = 10, reps = 50)

tunedK = tuneParams("classif.knn", task = irisTask,
                     resampling = cvForTuning,
                     par.set = knnParamSpace,
                     control = gridSearch)

tunedK
tunedK$x

knnTuningData = generateHyperParsEffectData(tunedK)

plotHyperParsEffect(knnTuningData, x = "k", y = "mmce.test.mean",
                    plot.type = "line") +
  theme_bw()

# make the learner with the tuned K
tunedKnn = setHyperPars(makeLearner("classif.knn"), par.vals = tunedK$x)

# re-train the knn model using the tuned K
tunedKnnModel = train(tunedKnn, irisTask)

# could create a tibble with new data and use the model to predict what
# species the new data would be classified as.

# 6 Cross-validate this iris kNN model using nested cross-validation, where the
# outer cross-validation is holdout with a two-thirds split:

inner = makeResampleDesc("CV")

outerHoldout = makeResampleDesc("Holdout", split = 2/3, stratify = TRUE)

knnWrapper = makeTuneWrapper("classif.knn", resampling = inner,
                              par.set = knnParamSpace,
                              control = gridSearch)

holdoutCVWithTuning = resample(knnWrapper, irisTask,
                                resampling = outerHoldout)
holdoutCVWithTuning

#7 Repeat the nested cross-validation using 5-fold,
# non-repeated cross-validation as the outer loop.
# Which of these methods gives you a more stable MMCE estimate
# when you repeat them?

outerKfold = makeResampleDesc("CV", iters = 5, stratify = TRUE)

kFoldCVWithTuning = resample(knnWrapper, irisTask,
                              resampling = outerKfold)

kFoldCVWithTuning

resample(knnWrapper, irisTask, resampling = outerKfold)

# Repeat each validation procedure 10 times and save the mmce value.

library(tidyverse)

kSamples = map_dbl(1:10, ~resample(
  knnWrapper, irisTask, resampling = outerKfold)$aggr
)

hSamples = map_dbl(1:10, ~resample(
  knnWrapper, irisTask, resampling = outerHoldout)$aggr
)

hist(kSamples, xlim = c(0, 0.11))

hist(hSamples, xlim = c(0, 0.11))
# Holdout CV introduces more variance.
