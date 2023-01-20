install.packages("tidyverse")
library(tidyverse)

myTib = tibble(x = 1:4,
               y = c("london","beijing","las vegas","berlin"))

myTib

data("starwars")
starwars

sequentialTib = tibble(nItems = c(12, 45, 107),
                        cost = c(0.5, 1.2, 1.8),
                        totalWorth = nItems * cost)

sequentialTib

######################
###Exercise 1#########
data("mtcars")
mtcars

mtcarsTib <- as_tibble(mtcars)

mtcarsTib

summary(mtcarsTib)
#####################

library(tibble)
library(dplyr)

data("CO2")
CO2tib = as_tibble(CO2)

CO2tib

selectedData = select(CO2tib, 1, 2, 3, 5)

selectedData

#####################
####Exercise 2#######

selectedDatamtcars = select(mtcarsTib, !7:8)

selectedDatamtcars

#####################

filteredData = filter(selectedData, uptake > 16)

filteredData

#####################
#####Exercise 3######

filteredDatamtcars = filter(mtcarsTib, cyl != 8)

filteredDatamtcars
#####################

groupedData = group_by(filteredData, Plant)

groupedData

summarisedData = summarise(groupedData, meanUp = mean(uptake),
                           sdUp = sd(uptake))

summarisedData

mutatedData = mutate(summarisedData, CV = (sdUp / meanUp) * 100)

mutatedData

arrangedData = arrange(mutatedData, CV)

arrangedData

arrangedData2 = arrange(mutatedData, desc(CV))

arrangedData2

#########################
####Concise version######
library(dplyr)

arrangedData = CO2tib %>%
  select(c(1:3, 5)) %>%
  filter(uptake > 16) %>%
  group_by(Plant) %>%
  summarize(meanUp = mean(uptake), sdUp = sd(uptake)) %>%
  mutate(CV = (sdUp / meanUp) * 100) %>%
  arrange(CV)

arrangedData


#########################
#####Exercise 4##########

library(dplyr)

arrangedmtcarData = mtcarsTib %>%
  group_by(gear) %>%
  summarize(medianmpg = median(mpg), mediandisp = median(disp)) %>%
  mutate(mpgBYdisp = medianmpg/mediandisp)

arrangedmtcarData

#########################

########ggplot###########

library(ggplot2)
data("iris")

myPlot = ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width)) +
  geom_point() +
  theme_linedraw()

myPlot

myPlot2 = ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width)) +
  geom_density_2d_filled() +
  geom_point() +
  geom_smooth() +
  theme_linedraw()

myPlot2

myPlot +
  geom_density_2d() +
  geom_smooth()

myPlot +
  geom_density_2d_filled() +
  geom_smooth()

ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, shape = Species)) +
  geom_point() +
  theme_linedraw()

ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, col = Species)) +
  geom_point() +
  theme_linedraw()

ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width)) +
  facet_wrap(~ Species) +
  geom_point() +
  theme_bw()

###############################
####Exercise 5#################


myPlot3 = ggplot(mtcarsTib, aes(x = drat, y = wt, col = carb)) +
  geom_point() +
  theme_bw()

myPlot3

myPlot4 = ggplot(mtcarsTib, aes(x = drat, y = wt, col = as.factor(carb))) +
  geom_point() +
  theme_bw()

myPlot4

#############################
##########TidyR##############

library(tibble)

library(tidyr)

patientData = tibble(Patient = c("A", "B", "C"),
                     Month0 = c(21, 17, 29),
                     Month3 = c(20, 21, 27),
                     Month6 = c(21, 22, 23))

patientData

tidyPatientData = gather(patientData, key = Month, value = BMI, -Patient)

tidyPatientData

#undo or make data wide
spread(tidyPatientData, key = Month, value = BMI)

#############################
####Exercise 6###############

tidymtcarsData = gather(mtcarsTib, key = vs_am_gear_carb, value = value,
                        -c(mpg, cyl, disp, hp, drat, wt, qsec))

tidymtcarsData

##############################
#####Purrr####################
library(purrr)

a = 20
pure = function(){
  a = a + 1
  a
}

side_effect = function(){
  a <<- a + 1
  a
}


c(pure(), pure())

c(side_effect(), side_effect())


listOfNumerics = list(a = rnorm(5),
                       b = rnorm(9),
                       c = rnorm(10))
listOfNumerics

elementLengths = vector("list", length = 3)
for(i in seq_along(listOfNumerics)) {
  elementLengths[[i]] = length(listOfNumerics[[i]])
}
elementLengths

#This code is difficult to read, requires us to predefine an empty vector
#to prevent the loop from being slow, and has a side effect:
#if we run the loop again, it will overwrite
#the elementLengths list.
#Instead, we can replace the for loop with the map() function.
#The first argument of all the functions in the map family is the data
#we’re iterating over. The second argument is the function we’re applying
#to each list element.

map(listOfNumerics, length)

#Returning atomic vectors. It is important to state the type of output
#to prevent bugs from unexpected types of output.

map_int(listOfNumerics, length)

map_chr(listOfNumerics, length)

map_lgl(listOfNumerics, length)

###########################
#####Exercise 7############

map_lgl(mtcars, ~sum(.) > 1000)
map_lgl(mtcars, function(.) sum(.) > 1000)


#Calculating the median of each column instead
map_lgl(mtcars, ~median(.) > 3)

###########################

map_df(listOfNumerics, length)

#using anonymous functions (base R) inside the map() family
# . is the anonymous function which is the element that map()
# is iterating over.
# purrr provides a shorthand for this and it is the tilde symbol
# ~

map(listOfNumerics, function(.) . + 2)

map(listOfNumerics, ~. + 2)

#Using walk for the side effects of a function, such as multiple plots

#Example
# splits the plotting device
#into two rows and four columns for base plots
par(mfrow = c(1, 3))

walk(listOfNumerics, hist)

#use the name of each list element as the title for its histogram
#.x to reference the list element we’re iterating over and
#.y to reference its name/index

iwalk(listOfNumerics, ~hist(.x, main = .y))

#Each of the map() functions has an i version that lets us reference
#each element’s name/index.

#Iterating over multiple lists simultaneously

multipliers = list(0.5, 10, 3)

map2(.x = listOfNumerics, .y = multipliers, ~.x * .y)

#start by using the expand.grid() function to create a data frame containing
# every combination of the input vectors. Because data frames are really
#just lists of columns, supplying one to pmap() will iterate a function
#over each column in the data frame.

arguments = expand.grid(n = c(100, 200),
                         mean = c(1, 10),
                         sd = c(1, 10))
arguments

#the function we ask pmap() to iterate over will be run using the
#arguments contained in each row of the data frame. Therefore, pmap()
#will return eight different random samples, one corresponding to each
#combination of arguments in the data frame.

par(mfrow = c(2, 4))

pmap(arguments, rnorm) %>%
  iwalk(~hist(.x, main = paste("Element", .y)))



