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

library(mlr)
library(tidyverse)

data(Zoo, package = "mlbench")
zooTib = as_tibble(Zoo)
zooTib

zooTib = mutate_if(zooTib, is.logical, as.factor)

zooTask = makeClassifTask(data = zooTib, target = "type")

tree = makeLearner("classif.rpart")

