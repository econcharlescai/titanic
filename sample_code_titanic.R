#1. import data
train <- read.csv("/Users/charlescai/Documents/titanic/all/train.csv")
submission <- read.csv("/Users/charlescai/Documents/titanic/all/test.csv")

#2. visualize data

#2.1 know the variables
str(train)
str(submission)

#2.2 find out if there're missing values
pacman::p_load(ggplot2)
is.na(train)


#3. clean data

#4. split train and test set

#5. different models and evaluation of models

#6. repeat 4 and 5, and choose the model

#7. produce the submission file