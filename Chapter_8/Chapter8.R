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

# Plot the decision tree
#install.packages("rpart.plot")
library(rpart.plot)

treeModelData = getLearnerModel(tunedTreeModel)

rpart.plot(treeModelData, roundint = FALSE,
           box.palette = "BuBn",
           type = 5)

printcp(treeModelData, digits = 3)
summary(treeModelData)

# Cross-validating the decision tree model
# Remember: you must include data-dependent pre-processing in your
# cross-validation procedure.

# Outer loop: 5-fold cross-validation.
# Inner loop: cvForTuning resampling
# Wrapper (Learner and hyperparameter tuning): inner cross-validation strategy
# (cvForTuning), hyperparameter space, and search method -> (makeTuneWrapper())
# Parallelise with parallelStartSocket() and start CV procress with resample()
# resample takes the following args: wrapped learner, task, outer CV strategy.

outer = makeResampleDesc("CV", iters = 5)

treeWrapper = makeTuneWrapper("classif.rpart", resampling = cvForTuning,
                              par.set = treeParamSpace,
                              control = randSearch)

parallelStartSocket(cpus = detectCores())

cvWithTuning = resample(treeWrapper, zooTask, resampling = outer)
parallelStop()

cvWithTuning

# The model has a tendency to overfit during cross-validation.
# How do we overcome this problem? The answer is to use an ensemble method.
