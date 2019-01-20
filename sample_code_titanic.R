setwd(getwd())

#1. import data
train_df <- read.csv("./all/train.csv")
eval_df <- read.csv("./all/test.csv")

#2. visualize data

#2.1 know the variables
str(train_df)
str(eval_df)

#2.2 find out if there're missing values
library(pacman)
pacman::p_load(ggplot2)
is.na(train)


#3. clean data

#4. split train and test set

#5. different models and evaluation of models

#6. repeat 4 and 5, and choose the model

#7. produce the submission file