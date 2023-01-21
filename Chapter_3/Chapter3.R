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
