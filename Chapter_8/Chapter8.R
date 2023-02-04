#### Ensemble methods - Chapter 8 ####

# Three ensemble methods:
# Bootsrap aggregating
# Boosting
# Stacking
# Invariably these are all used in order to reduce variance by generating
# multiple models that classify a given variable(s) of interest.

#### Bootstrap aggregating (or bagging) ####
# It's used for random forest classifier algorithms.
# The premise is simple:
# 1 Decide how many sub-models you’re going to train (training happens
# in parallel).
# 2 For each sub-model, randomly sample cases from the training set,
# with replacement, until you have a sample the same size as the original
# training set.
# 3 Train a sub-model on each sample of cases.
# 4 Pass new data through each sub-model, and let them vote on the prediction.
# 5 The modal prediction (the most frequent prediction) from all the
# sub-models is used as the predicted output.

# The most critical part of bagging is the random sampling of the cases.

# An important feature of bootstrap aggregating/random forest is that at
# each node of a particular tree, the algorithm randomly selects a proportion
# of the predictor variables it will consider for that split. At the next node,
# the algorithm makes another random selection of predictor variables it will
# consider for that split, and so on. While this may seem counterintuitive,
# the result of randomly sampling cases and randomly sampling features is to
# create individual trees that are highly uncorrelated.
# Trees that contain the same splits as each other don’t contribute any more
# information. This is why it’s desirable to have uncorrelated trees, so that
# different trees contribute different predictive information. Randomly
# sampling cases reduces the impact that noise and outlying cases have
# on the model.

#### Boosting ####
# Two main methods: adaptive and gradient boosting.
# Boosting trains many individual models, but builds them sequentially. Each
# additional model seeks to correct the mistakes of the previous ensemble
# of models.
# Boosting is most beneficial when using weak learners as the submodels. For
# this reason, boosting has been traditionally applied to shallow decision
# trees.
# The function of boosting is to combine many weak learners together to form
# one strong ensemble learner. The reason we use weak learners is that there
# is no improvement in model performance when boosting with strong learners
# versus weak learners. So why waste computational resources training hundreds
# of strong, probably more complex learners, when we can get the same
# performance by training weak, less complex ones?

# model weight and case weight calculation in adaptive boosting
# model weight = 0.5 x ln(1-p(incorrect)/p(correct))
# case weight = initial weight x e^-model weight (if correctly classified)
# case weight = initial weight x e^model weight (if incorrectly classified)

# Gradient boosting uses a different method to deal with learning from previous
# models. Rather than weighing the cases differently depending on the
# accuracy of their classification, subsequent models try to predict
# the residuals of the previous ensemble of models.
# We can quantify the residual error of a classification model as:
# - The proportion of all cases incorrectly classified
# - The log loss (penalises a model that makes incorrect classifications
# confidently. i.e. an overly confident/cocky model).

# By minimizing the residual error, subsequent models will, in effect, favor
# the correct classification of cases that were previously misclassified
# (thereby modeling the residuals).

# Calculating the log loss
# log loss = -1/N*sum(N, i=1)*sum(K, k=1)*Y(ik)*ln(p(ik))
# N is the number of cases
# K is the number of classes
# ln is the natural log
# Y(ik) is an indicator as to whether label k is the correct classification
# for case i.
# p(ik) is the proportion of cases belonging to the same class as case i that
# were correctly classified.
# 1) For every case in the training set:
# a) Take the proportion of cases belonging to the same class as the cases that
# were correctly classified.
# b) Take the natural log of these proportions.
# 2) Sum these logs.
# 3) Multiply by -1/N

# Gradient boosting doesn’t necessarily train sub-models on samples of the
# training set. If we choose to sample the training set, the process is
# called stochastic gradient boosting. Sampling in stochastic gradient descent
# is usually without replacement, which means it isn’t a bootstrap sample.

# XGBoost is an implementation of gradient boosting. It is capable of the
# following:
# * It can build different branches of each tree in parallel, speeding up model
# building.
# * It can handle missing data.
# * It employs regularization, which prevents individual predictors from
# having too large of an impact on predictions (helps to prevent overfitting).


# The strengths of the random forest and XGBoost algorithms are as follows:
# * They can handle categorical and continuous predictor variables
# (though XGBoost requires some numerical encoding).
# * They make no assumptions about the distribution of the predictor variables.
# * They can handle missing values in sensible ways.
# * They can handle continuous variables on different scales.
# * Ensemble techniques can drastically improve model performance over
# individual trees. XGBoost in particular is excellent at reducing both bias
# and variance.
# The weaknesses of tree-based algorithms are these:
# * Random forest reduces variance compared to rpart but does not reduce bias
# (XGBoost reduces both).
# XGBoost can be computationally expensive to tune because it has many
# hyperparameters and grows trees sequentially.

#### Stacking ####
# Learning from the predictions made by other models
#  In bagging and boosting, the learners are often (but don’t always have to be)
# homogeneous. Put another way, all of the sub-models were learned by the
# same algorithm (decision trees). Stacking explicitly uses different
# algorithms to learn the sub-models.

# The idea behind stacking is that we create base models that are good at
# learning different patterns in the feature space. One model may then be good
# at predicting in one area of the feature space but makes mistakes in another
# area. One of the other models may do a good job of predicting values in an
# area of the feature space where the others do poorly. So here’s the key in
# stacking: the predictions made by the base models are used as predictor
# variables (along with all the original predictors) by another model:
# the stacked model.



####run code from 7.1 to 7.8 to be used for chapter 8 ####
library(mlr)
library(tidyverse)

data(Zoo, package = "mlbench")
zooTib = as_tibble(Zoo)
zooTib

# mlr doesn't allow generation of tasks with logical predictors, so they
# must be converted to factors.

zooTib = mutate_if(zooTib, is.logical, as.factor)
map_dbl(zooTib, ~sum(is.na(.)))

# train the decision tree model

zooTask = makeClassifTask(data = zooTib, target = "type")

tree = makeLearner("classif.rpart")

# Printing available rpart hyperparameters and then define hyperparameter
# space for tuning (random search instead of grid search and fold CV without
# stratification because there are not enough cases to stratify)

getParamSet(tree)

treeParamSpace = makeParamSet(
  makeIntegerParam("minsplit", lower = 5, upper = 20),
  makeIntegerParam("minbucket", lower = 3, upper = 10),
  makeNumericParam("cp", lower = 0.01, upper = 0.1),
  makeIntegerParam("maxdepth", lower = 3, upper = 10))

randSearch = makeTuneControlRandom(maxit = 200)

cvForTuning = makeResampleDesc("CV", iters = 5)

# Parallelise hyperparameter tuning to speed things up
# Use tuneParams tostart tuning process
# arguments are: learner (tree),
# task (task = zooTask),
# cv method (resampling = cvForTuning),
# hyperparameter space (par.set = treeParamSpace), and
# search method (control = randSearch)
library(parallel)
library(parallelMap)

parallelStartSocket(cpus = detectCores())
tunedTreePars = tuneParams(tree, task = zooTask,
                           resampling = cvForTuning,
                           par.set = treeParamSpace,
                           control = randSearch)
parallelStop()
tunedTreePars

# Training the final tuned model
# create a learner with the tuned hyperparameter
# contained in tunedTreePars$x

tunedTree = setHyperPars(tree, par.vals = tunedTreePars$x)

# train the final model

tunedTreeModel = train(tunedTree, zooTask)

#### Building a random forest model####
# Compare vs the performance of the rpart.classif

# There are four important hyperparameters for us to consider:
# * ntree: The number of individual trees in the forest.
# * mtry: The number of features to randomly sample at each node.
# * nodesize: The minimum number of cases allowed in a leaf (the same as
# minbucket in rpart).
# * maxnodes: The maximum number of leaves allowed.

# There is no downside to having more trees aside from computational cost: at
# some point, we get diminishing returns. Rather than tuning this value, fix it
# to a number of trees that fits your computational budget, generally several
# hundred to the low thousands is small/conservative (this can be optimised and
# tested to get a reasonable trade-off). The other three parameters do need to
# be tuned though.

# First thing to do is to make a learner.
# (cont. from 7.8 on previous chapter using zooTask)

forest = makeLearner("classif.randomForest")

# Create hyperparameter tuning space
forestParamSpace = makeParamSet(
  makeIntegerParam("ntree", lower = 300, upper = 300),
  makeIntegerParam("mtry", lower = 6, upper = 12),
  makeIntegerParam("nodesize", lower = 1, upper = 5),
  makeIntegerParam("maxnodes", lower = 5, upper = 20))

# Defines random search method with 100 iterations
randSearch = makeTuneControlRandom(maxit = 100)

# Defines a 5-fold cross-validation strategy
cvForTuning = makeResampleDesc("CV", iters = 5)

parallelStartSocket(cpus = detectCores())

# Tunes the hyperparameters
tunedForestPars <- tuneParams(forest, task = zooTask,
                              resampling = cvForTuning,
                              par.set = forestParamSpace,
                              control = randSearch)
parallelStop()

tunedForestPars

# Train the final model

tunedForest = setHyperPars(forest, par.vals = tunedForestPars$x)

tunedForestModel = train(tunedForest, zooTask)

# Plotting the out of the bag error

forestModelData = getLearnerModel(tunedForestModel)

species = colnames(forestModelData$err.rate)

plot(forestModelData, col = 1:length(species), lty = 1:length(species))
legend("topright", species,
       col = 1:length(species),
       lty = 1:length(species))

# How to interpret the out of the bag error (OOB): If you train a model
# and the mean out-of-bag error doesn’t stabilize, you should add
# more trees (that simple).

# Cross-validate the model (including hyperparameter tuning)

# Outer loop (five-fold CV)
outer = makeResampleDesc("CV", iters = 5)

# make the wrapper using the usual arguments, classifier, hyperparameter
# tuning (inner loop), parameter space, and control search strategy)

forestWrapper = makeTuneWrapper("classif.randomForest",
                                 resampling = cvForTuning,
                                 par.set = forestParamSpace,
                                 control = randSearch)

parallelStartSocket(cpus = detectCores())

cvWithTuning = resample(forestWrapper, zooTask, resampling = outer)

parallelStop()
cvWithTuning

#### Same problem tackled with XGBoost ####

# Eight hyperparameters:
# eta: learning rate (between 0 and 1). Model weights are multiplied by this
# value. If set low it slows training but it helps prevent overfitting.

# gamma: the minimum amount of node splitting needed to improve predictions.
# similar to the cp value in rpart.

# max_depth: max level to which each tree can grow.

# min_child_weight: minimum degree of impurity needed in a node before
# attempting to split it.

# subsample: proportion of cases to be randomly sampled (without replacement)
# for each tree (if set to 1 it uses all cases in the training set).

# colsample_bytree: proportion of predictor variables sampled for each tree.
# can also sample by level or node with colsample_bylevel and colsample_bynode.

# nrounds: number of sequentially built trees in the model.

# eval_metric: type of residual error/loss function to be used. Either proportion
# of cases that were incorrectly classified (merror) or the log loss (mlogloss)

### Create a learner with classification method as XGBoost.

xgb = makeLearner("classif.xgboost")

# mutate all non numerical variables. Leave type as categorical
# by using the .vars argument and selecting all but type
# you can then use .funs argument to specify the mutation into numeric

zooXgb = mutate_at(zooTib, .vars = vars(-type), .funs = as.numeric)

xgbTask = makeClassifTask(data = zooXgb, target = "type")

# Most of our predictors are binary except legs, which makes sense as a
# numeric variable. However, if we have a factor with many discrete levels, it
# doesn't make sense to treat it as numeric but it can work well if you
# recode each level of the factor as an arbitrary integer and let the
# decision tree find the best split for us. This is called numerical encoding
# (it's what has been done to the variables in the dataset).
# You may have heard of another method of encoding categorical features
# called one-hot encoding. One-hot encoding factors for tree-based models
# often results in poor performance.

# Define the hyperparameter space for tuning

xgbParamSpace = makeParamSet(
  makeNumericParam("eta", lower = 0, upper = 1),
  makeNumericParam("gamma", lower = 0, upper = 5),
  makeIntegerParam("max_depth", lower = 1, upper = 5),
  makeNumericParam("min_child_weight", lower = 1, upper = 10),
  makeNumericParam("subsample", lower = 0.5, upper = 1),
  makeNumericParam("colsample_bytree", lower = 0.5, upper = 1),
  makeIntegerParam("nrounds", lower = 30, upper = 50),
  makeDiscreteParam("eval_metric", values = c("merror", "mlogloss")))

# Define the search method and number of iterations
randSearch = makeTuneControlRandom(maxit = 10000)

# Set cross-validation strategy (5-fold cross-validation)
cvForTuning = makeResampleDesc("CV", iters = 5)

tunedXgbPars = tuneParams(xgb, task = xgbTask,
                           resampling = cvForTuning,
                           par.set = xgbParamSpace,
                           control = randSearch)
tunedXgbPars

# Train the final XGBoost model using the tuned hyperparameters.
# First make a learner with setHyperPars() and then pass it to the train()
# function.

tunedXgb = setHyperPars(xgb, par.vals = tunedXgbPars$x)

tunedXgbModel = train(tunedXgb, xgbTask)

# Plot the loss function against the iteration number to see if we have
# included enough trees.

xgbModelData = getLearnerModel(tunedXgbModel)

# If the hyperparameter tuning selects log loss instead of classification error
# you just change the argument in aes(iter, train merror) to aes(iter, mlogloss)
# and vice versa.

ggplot(xgbModelData$evaluation_log, aes(iter, train_merror)) +
  geom_line() +
  geom_point()

# Plotting the individual decision trees
#install.packages("DiagrammeR")
library(DiagrammeR)

xgboost::xgb.plot.tree(model = xgbModelData, trees = 1:5)

# represent the final ensemble as a single tree structure
xgboost::xgb.plot.multi.trees(xgbModelData)

# Cross-validate the model-building process

# outer loop
outer = makeResampleDesc("CV", iters = 3)

# wrapper
xgbWrapper = makeTuneWrapper("classif.xgboost",
                              resampling = cvForTuning,
                              par.set = xgbParamSpace,
                              control = randSearch)
# inner loop
cvWithTuning = resample(xgbWrapper, xgbTask, resampling = outer)

cvWithTuning

# Benchmarking vs other algos
learners = list(makeLearner("classif.knn"),
                makeLearner("classif.LiblineaRL1LogReg"),
                makeLearner("classif.svm"),
                tunedTree,
                tunedForest,
                tunedXgb)
benchCV = makeResampleDesc("RepCV", folds = 10, reps = 5)
bench = benchmark(learners, xgbTask, benchCV)
