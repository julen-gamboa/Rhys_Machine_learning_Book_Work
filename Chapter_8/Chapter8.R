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
# Tuning the random forest hyperparameters (cont. from 7.8 on previous chapter)

forest = makeLearner("classif.randomForest")

# Creates hyperparameter tuning space)
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
