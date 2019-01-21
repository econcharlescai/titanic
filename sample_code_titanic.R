# This sample code takes a relatively simple way to tackle the kaggle competition: Titanic: Machine Learning from Disaster. (https://www.kaggle.com/c/titanic).
# As one of my early experiences in R, the specific methods used here are not very sophisticated. Nonetheless, I hope this can show my coding habit.
# Also, it might be worth mentioning here that I am using R for a machine learning class and will surely develop R skills as well as machine learning skills along the way.

# Prepared by Charles Cai 
# Last modified Jan 20, 2019

#0. libraries and setwd
library(pacman)
setwd(getwd())

#1. import data
train_df <- read.csv("./all/train.csv", na.strings = '')
test_df <- read.csv("./all/test.csv", na.strings = '')

#2. clean the data

#2.1 get familiar with the dataset
str(train_df)
str(test_df)

train_df$Pclass <- factor(train_df$Pclass)
test_df$Pclass <- factor(test_df$Pclass)

#2.2 find out if there're missing values
pacman::p_load(Amelia)
missmap(train_df)
missmap(test_df)

#2.3 fill in the NA's
# (note that in this exercise I will not use information in Cabin, so only need to deal with NA's in Age, Fare and Embarked.)
pacman::p_load(dplyr)
train_df <- train_df %>% subset(select = -c(Cabin, Name, Ticket))
test_df <- test_df %>% subset(select = -c(Cabin, Name, Ticket))

#2.3.1 To predict Age, use Fare, Parch, Sibsp, Sex, Pclass. Method: linear regression. If predicted Age is negative, set it to be 1.

age_df <- train_df %>% filter(!is.na(Age))
fit <- lm(Age ~ Fare + Embarked + Parch + SibSp + Sex + Pclass, data = age_df)
train_df <- train_df %>%
  mutate(Age = ifelse(is.na(Age),predict(fit, .),Age)) %>%
  mutate(Age = ifelse(Age <= 0, 1, Age))
rm(age_df)
rm(fit)

age_df <- test_df %>% filter(!is.na(Age))
fit <- lm(Age ~ Fare + Embarked + Parch + SibSp + Sex + Pclass, data = age_df)
test_df <- test_df %>%
  mutate(Age = ifelse(is.na(Age),predict(fit, .),Age)) %>%
  mutate(Age = ifelse(Age <= 0, 1, Age))
rm(age_df)
rm(fit)

#2.3.2 To predict Embarked, use Age, Fare, Sex, Pclass. Method: multinomial regression.
pacman::p_load(nnet)
embark_df <- train_df %>% filter(!is.na(Embarked)) %>% subset(select = c(Age, Fare, Sex, Pclass, Embarked))
mul <- multinom(Embarked ~ ., data = embark_df)
train_df <- train_df %>%
  mutate(Embarked = ifelse(is.na(Embarked), predict(mul,.), Embarked))
rm(embark_df)
rm(mul)

#2.3.3 To predict Fare, use Age, Embarked, Pclass. Methor: linear regression.
fare_df <- test_df %>% filter(!is.na(Age))
fit <- lm(Fare ~ Age + Embarked + Pclass, data = fare_df)
test_df <- test_df %>%
  mutate(Fare = ifelse(is.na(Fare),predict(fit, .),Fare))
rm(fare_df)
rm(fit)

#2.4 Create Agebin variable for later use
train_df$Agebin <- cut(train_df$Age, seq(0,80,10))
test_df$Agebin <- cut(test_df$Age, seq(0,80,10))

#2.5 Change Embarked into factor variable
train_df$Embarked <- train_df$Embarked %>% as.factor()
test_df$Embarked <- test_df$Embarked %>% as.factor()

#2.6 Merge Parch and Sibsp
train_df$Fam <- train_df$Parch + train_df$SibSp
train_df <- train_df %>%
  subset(select=-c(Parch, SibSp))
test_df$Fam <- test_df$Parch + test_df$SibSp
test_df <- test_df %>%
  subset(select=-c(Parch, SibSp))

#3. visualize the data

#3.1 see patterns of survival (Sex, Embarked, Pclass, Parch, Sibsp)
pacman::p_load(ggplot2)
survive_plot <- ggplot(data = train_df, aes(x = Survived))
survive_plot + geom_bar()
survive_plot + geom_bar(aes(fill = Sex), position = "dodge") + scale_fill_brewer(palette = "Blues")
survive_plot + geom_bar(aes(fill = as.factor(Pclass)), position = "dodge")

ggplot(data = train_df, aes(x = Pclass, y = Survived, group = Sex)) + 
  stat_summary(aes(color = Sex), fun.y = "mean", geom = "point") + 
  stat_summary(aes(color = Sex), fun.y = "mean", geom = "line")
ggplot(data = train_df, aes(y = Survived)) + 
  stat_summary(aes(x = Embarked, color = Sex), fun.y = "mean", geom = "point")
ggplot(data = train_df, aes(x = Agebin, y = Survived, group = Sex)) + 
  stat_summary(aes(color = Sex), fun.y = "mean", geom = "point") + 
  stat_summary(aes(color = Sex), fun.y = "mean", geom = "line")


#4. split train and test set
# use 70% for train and 30% for test
train_size <- floor(0.7*nrow(train_df))
pacman::p_load(e1071)
#5. different models and evaluation of models
#5.1 logistic regression
pacman::p_load(caret)

LogitAr <- 0
set.seed(123) 

for (i in 1:10){
  trainrows <- sample(seq_len(nrow(train_df)), size = train_size)
  train_set <- train_df[trainrows,]
  test_set <- train_df[-trainrows,]

  logreg <- glm(Survived ~ . - PassengerId - Age, data = train_set)
  Logit <- predict.glm(logreg, newdata = test_set, type = "response")
  Logit <- ifelse(Logit > 0.5, 1, 0)
  cm <- confusionMatrix(table(Logit, test_set$Survived))
  LogitAr <- LogitAr + cm$overall['Accuracy']
}  
LogitAr <- LogitAr / 10

#5.2 Gaussian Naive Bayes

GnbAr <- 0
set.seed(123)

for (i in 1:10){
  trainrows <- sample(seq_len(nrow(train_df)), size = train_size)
  train_set <- train_df[trainrows, ]
  test_set <- train_df[-trainrows, ]
  gnb <- naiveBayes(as.factor(Survived) ~ Pclass + Sex + Agebin, data = train_set)
  Gnb <- predict(gnb, newdata = test_set)
  cm <- confusionMatrix(table(Gnb, test_set$Survived))
  GnbAr <- GnbAr + cm$overall['Accuracy']
}  
GnbAr <- GnbAr / 10
  
#5.3 KNN
pacman::p_load(class)
KnnAr <- 0
set.seed(123) 

for (i in 1:10){
  trainrows <- sample(seq_len(nrow(train_df)), size = train_size)
  train_set <- train_df[trainrows, ]
  test_set <- train_df[-trainrows, ]
  train_knn <- subset(train_set, select = c(Survived, Age, Fare, Pclass, Fam, Sex))
  test_knn <- subset(test_set, select = c(Survived, Age, Fare, Pclass, Fam, Sex))

  train_knn$Sex <- ifelse(train_knn$Sex == "male", 1, 0)
  train_knn$Pclass <- train_knn$Pclass %>% 
    as.numeric()

  test_knn$Sex <- ifelse(test_knn$Sex == "male", 1, 0)
  test_knn$Pclass <- test_knn$Pclass %>% 
    as.numeric()
  train_knn$Survived <- train_knn$Survived %>% as.factor()
  test_knn$Survived <- test_knn$Survived %>% as.factor()

  Prdknn <- knn(train = train_knn, test = test_knn, cl = train_knn$Survived, k=5)
  cm <- confusionMatrix(table(Prdknn, test_set$Survived))
  KnnAr <- KnnAr + cm$overall['Accuracy']
}  
KnnAr <- KnnAr / 10


#5.4 decision tree
pacman::p_load(tree)

TreeAr <- 0
set.seed(123) 

for (i in 1:10){
  trainrows <- sample(seq_len(nrow(train_df)), size = train_size)
  train_set <- train_df[trainrows,]
  test_set <- train_df[-trainrows,]
  train_tr <- train_set
  test_tr <- test_set
  train_tr$Survived <- train_tr$Survived %>% as.factor()
  test_tr$Survived <- test_tr$Survived %>% as.factor()

  tr <- tree(Survived ~ . - PassengerId - Age, data = train_tr )
  Tr <- predict(tr, test_tr, type = "class")
  cm <- confusionMatrix(table(Tr, test_set$Survived))
  TreeAr <- TreeAr + cm$overall['Accuracy']
}  
TreeAr <- TreeAr / 10


#5.5 random forest 
pacman::p_load(randomForest)
RfAr <- 0
set.seed(123) 
for (i in 1:10){
  trainrows <- sample(seq_len(nrow(train_df)), size = train_size)
  train_set <- train_df[trainrows,]
  test_set <- train_df[-trainrows,]
  rf <- randomForest(as.factor(Survived) ~ . - PassengerId - Agebin, data = train_set, ntree = 1000, mtry = 2, importance = TRUE)
  Rf <- predict(rf, test_set, type = "class")
  cm <- confusionMatrix(table(Rf, test_set$Survived))
  RfAr <- RfAr + cm$overall['Accuracy']
}  
RfAr <- RfAr / 10

#6. choose the best models from 5 (random forest in this case)
ar_df <- as.data.frame(t(data.frame(LogitAr, GnbAr, KnnAr, TreeAr, RfAr)))
ar_df$Model <- rownames(ar_df)
ggplot(data=ar_df, aes(x = Model, y = Accuracy)) + geom_col()


#7. produce the submission file
test_df$Embarked <- test_df$Embarked %>% as.numeric() %>% as.factor() 
rf <- randomForest(as.factor(Survived) ~ . - PassengerId - Agebin, data = train_df, ntree = 1000, mtry = 2, importance = TRUE)
test_df$Survived <- predict(rf, test_df, type = "class")
submission <- subset(test_df, select = c(PassengerId,Survived))
write.csv(submission, 'submission.csv', row.names=FALSE)
