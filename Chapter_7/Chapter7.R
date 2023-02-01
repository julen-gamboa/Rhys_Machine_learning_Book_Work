# The intuition behind tree-building is quite simple, and each individual
# tree is very interpretable.
# * It can handle categorical and continuous predictor variables.
# * It makes no assumptions about the distribution of the predictor variables.
# * It can handle missing values in sensible ways.
# * It can handle continuous variables on different scales.
# * Individual trees are very susceptible to overfitting—so much so that
# they are rarely used.

# At each stage of the tree-building process, the rpart algorithm considers
# all of the predictor variables and selects the predictor that does the best
# job of discriminating the classes. It starts at the root and then,
# at each branch, looks again for the next feature that will best discriminate
# the classes of the cases that took that branch. But how does rpart decide
# on the best feature at each split? This can be done a few different ways,
# and rpart offers two approaches: the difference in entropy
# (called the information gain) and the difference in Gini index
# (called the Gini gain).
# If you’re concerned that you’re missing the best-performing model, you
# can always compare Gini index and entropy during hyperparameter tuning.

# Entropy and the Gini index are two ways of trying to measure the same
# thing: impurity. Impurity is a measure of how heterogeneous the classes
# are within a node.


# The Gini index of any node is calculated as follows:
# Gini index = 1 -(p(A)^2 + p(B)^2)
# Gini index(split) = p(left) * Gini index(left) + p(right) * Gini index(right)
# Gini gain = Gini index(parent node) - Gini index(split)

#  The Gini gain at a particular node is calculated for each predictor
# variable, and the predictor that generates the largest Gini gain is
# used to split that node. This process is repeated for every node as
# the tree grows.

# rpart is a greedy algorithm
# * The algorithm isn’t guaranteed to learn a globally optimal model.
# * If left unchecked, the tree will continue to grow deeper until
# all the leaves are pure (of only one class).
# *  For large datasets, growing extremely deep trees becomes computationally
# expensive.

# Dealing with expensive tree building involves:
# * Growing a full tree, and then pruning it.
# * Employ stopping criteria (preferred).

# Stopping criteria (you wanna tune these)
# * Minimum number of cases in a node before splitting (minsplit)
# * Maximum depth of the tree (maxdepth)
# * Minimum improvement in performance for a split (cp)
# * Minimum number of cases in a leaf (minbucket)

# Other hyperparameters
# maxcompete: controls how many candidate splits can be displayed for each node
# in the model summary. It is useful to understand what the next-best split was
# after the one that was used, it does not affect model performance, just the
# summary.
# maxsurrogate: controls how many surrogate splits are shown. Surrogate splits
# are splits used if a particular case is missing data for that split. This
# allows rpart to handle missing data while it learns which splits can be
# used in place of N.A. variables. This hyperparameter controls how many of
# these surrogates to retain in the model.
# usesurrogate: controls how the algorithm uses surrogate splits.
# 0 = no surrogates used, therefore cases with missing data won't be
# classified.
# 1 = surrogates are used but if there is missing data for the split and
# all other surrogate splits, then the case will not be classified.
# 2 = surrogates will be used but when there is a case with missing data for
# the actual split and for all surrogate splits, then it will be sent down
# the branch that contained most cases (most appropriate value is therefore 2)

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

