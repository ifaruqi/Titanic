

############### install and load packages ###############

#install.packages("tidyverse")
library(tidyverse)
library(data.table)
library(dplyr)

# Load caTools package for data partitioning
library(caTools)

# load Caret package for computing Confusion matrix
library(caret) 
library(mltools)

# load pROC package for ROC chart
library(pROC) 

# Packages for SVM and Random Forest
library(e1071)
library(randomForest)

# Packages for Decision Tree and LDA
library(party)
library(MASS)
library(tree)
library(rpart)

# Package for Gain
library(CustomerScoringMetrics)


############### Load the data ###############

# Load the data
titanic <- read_csv("train_and_test2.csv")
summary(titanic)
str(titanic)

############### Clean the data ###############

# Remove column zero
zerocol <- colnames(titanic[,grep("zero", names(titanic))])
titanic[zerocol] <- NULL


# Remove Passenger ID as this is not important in the analysis
titanic$Passengerid = NULL

# Change column name 
names(titanic)[names(titanic) == '2urvived'] <- 'survived'

# Change data type for certain variable to factor
facol <- c("Sex","sibsp", "Parch", "Pclass", "Embarked", "survived")

titanic[facol] <- lapply(titanic[facol], factor)

# Check cleaned data
str(titanic)
summary(titanic)

# Remove 2 NA data from Embarked
titanic <- na.omit(titanic)

# Check level of the target variable
levels(titanic$survived)

#log normalization & boxplot to catch outliers
boxplot(log(titanic$Age))$out
boxplot(log(titanic$Fare))$out

# Otlier seems to be present in the age column, however the numbers are still make sense so I will not remove the observation


############### Data Analysis ###############

# Apply hot encoding for all factor column except for target variable 
titanic<-one_hot(as.data.table(titanic), cols = c("Sex","sibsp", "Parch", "Pclass", "Embarked"))

# Split data into training and test data
# Set seed to 123
set.seed(123)

# Partition the data
#split = sample.split(titanic$survived, SplitRatio = 0.7) 
split = sample.split(titanic$survived, SplitRatio = 0.7) 

# Generate training and test sets and save as trainingset and testset
#trainingset = subset(titanic, split == TRUE) 
#testset = subset(titanic, split == FALSE) 

trainingset = subset(titanic, split == TRUE) 
testset = subset(titanic, split == FALSE) 

# Checking the proportionality of the target variable in the training data
table(trainingset$survived)
prop.table(table(trainingset$survived))


############### Logistic Regression ###############

# Build a linear model
LogReg <- glm(survived ~. , data = trainingset, family = "binomial")

# Predict the class probabilities of the test data
LogReg_pred <- predict(LogReg, testset, type="response")


# Predict the class 
LOGREG_survived <- ifelse(LogReg_pred > 0.5, "1", "0")

# Save the predictions as factor variables
LOGREG_survived <- as.factor(LOGREG_survived)

# Mode precision recall is choosen because we want to focus on the true positive rate
confusionMatrix(LOGREG_survived, testset$survived, positive='1', mode = "prec_recall")

############### Support Vector Machine ###############

# Build SVM model and assign it to SVM_model
SVM_model <- svm(survived ~. , data = trainingset, kernel= "radial", scale = TRUE, probability = TRUE)

# Predict the class of the test data
SVM_pred <- predict(SVM_model, testset)

# Use confusionMatrix to print the performance of SVM model
confusionMatrix(SVM_pred, testset$survived, positive = '1', mode = "prec_recall")

# SVM Tuning
grid_radial <- expand.grid(sigma = c(0.03704704),
                           C = c(1))

trctrl <- trainControl(method = "repeatedcv", number = 2, repeats = 2)
set.seed(3233)

svm_tune <- train(survived~Age+Fare+Sex_0+Sex_1+sibsp_0+sibsp_1+sibsp_2+sibsp_3+sibsp_4+sibsp_5+sibsp_8+Parch_0+Parch_1+Parch_2+Parch_3+Parch_4+Parch_5+Parch_6+Parch_9+Pclass_1+Pclass_2+Pclass_3+Embarked_0+Embarked_1+Embarked_2,
                  data = trainingset, method = "svmRadial",
                  trControl= trctrl,
                  tuneGrid = grid_radial,
                  allowParallel = T)

SVM_pred_tune <- predict(svm_tune, testset)

# Use confusionMatrix to print the performance of SVM model
confusionMatrix(SVM_pred_tune, testset$survived, positive = '1', mode = "prec_recall")

############### Decison Tree ###############

#Build a decision tree
decTree  <- ctree(survived ~., data = trainingset)

summary(decTree)

decTree_predict = predict(decTree, testset, type= "response")

# Confusion matrix
confusionMatrix(decTree_predict, testset$survived, positive='1', mode = "prec_recall")

############### Random Forest ###############

# Set random seed
set.seed(123)

# Build Random Forest model and assign it to RF_model
RF_model <- randomForest(survived ~., trainingset, ntree = 3000, mtry = 4)

# Print
print(RF_model)

importance(RF_model)

# Predict the class of the test data
RF_pred <- predict(RF_model, testset)

# Confusion matrix
confusionMatrix(RF_pred, testset$survived, positive='1', mode = "prec_recall")

# Random Forest Tuning
# Grid Tuning for mtry: Number of variables randomly sampled as candidates at each split.
set.seed(123)
control <- trainControl(trim=TRUE,method="repeatedcv", number = 3, repeats = 2)
tunegrid <- expand.grid(.mtry=c(2,4,8,16))
rf_gridsearch <- train(survived~Age+Fare+Sex_0+Sex_1+sibsp_0+sibsp_1+sibsp_2+sibsp_3+sibsp_4+sibsp_5+sibsp_8+Parch_0+Parch_1+Parch_2+Parch_3+Parch_4+Parch_5+Parch_6+Parch_9+Pclass_1+Pclass_2+Pclass_3+Embarked_0+Embarked_1+Embarked_2, 
                       data=trainingset, method="rf", metric="Accuracy", trControl=control,
                       allowParallel = TRUE,
                       tuneGrid = tunegrid)
print(rf_gridsearch)
plot(rf_gridsearch)

# Optimal mtry is 4

############### Visualization for Performance Analysis ###############

# Obtain class probabilities by using predict() and adding type = "prob" for Random Forest
RF_prob <- predict(RF_model, testset, type = "prob")  # Check the output for churn probabilties

DT_prob <- predict(decTree, testset, type = "prob")

SVM_pred <- predict(SVM_model, testset, probability = TRUE)

# Add probability = TRUE for SVM
SVM_prob <- attr(SVM_pred, "probabilities")  # Check the output for churn probabilties


# Logistic Regression
ROC_LogReg <- roc(testset$survived, LogReg_pred)

# Random Forest
ROC_RF <- roc(testset$survived, RF_prob[,2])

# Decision Tree
DT_prob_df <- as.data.frame(t(matrix(unlist(DT_prob), ncol=392)))
ROC_DT <- roc(testset$survived, DT_prob_df[,2])

# SVM
ROC_SVM <- roc(testset$survived, SVM_prob[,2])

# Plot the ROC curve for Logistic Regression, SVM and Random Forest
ggroc(list(LogReg = ROC_LogReg, SVM = ROC_SVM, DT = ROC_DT, RF = ROC_RF), legacy.axes=TRUE)+ xlab("FPR") + ylab("TPR") +
  geom_abline(intercept = 0, slope = 1, color = "darkgrey", linetype = "dashed")


#Calculate the area under the curve (AUC) for Logistic Regression 
auc(ROC_LogReg)

#Calculate the area under the curve (AUC) for SVM 
auc(ROC_SVM)

auc(ROC_DT)

#Calculate the area under the curve (AUC) for Random Forest 
auc(ROC_RF)

# Obtain cumulative gains table for Logistic Regression
GainTable_LogReg <- cumGainsTable(LogReg_pred, testset$survived, resolution = 1/100)

# Obtain cumulative gains table for SVM
GainTable_SVM <- cumGainsTable(SVM_prob[,2], testset$survived, resolution = 1/100)

# Obtain cumulative gains table for Random Forest
GainTable_RF <- cumGainsTable(RF_prob[,2], testset$survived, resolution = 1/100)

# Obtain cumulative gains table for Decision Tree
GainTable_DT <- cumGainsTable(DT_prob_df[,2], testset$survived, resolution = 1/100)

# Plot the gain chart

plot(GainTable_LogReg[,4], col="red", type="l",    
     xlab="Percentage of test instances", ylab="Percentage of correct predictions")
lines(GainTable_SVM[,4], col="blue", type ="l")
lines(GainTable_DT[,4], col="black", type ="l")
lines(GainTable_RF[,4], col="green", type ="l")
grid(NULL, lwd = 1)

legend("bottomright",
       c("LogReg", "SVM", "Random Forest", "Decision Tree"),
       fill=c("red","blue", "green", "black"))
