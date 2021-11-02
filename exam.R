library(lattice)
library(caret)
library(kernlab)
library(rattle)


training <- read.csv("~/pml-training.csv")
testing <- read.csv("~/pml-testing.csv")
set.seed(12345)


training <- training[,colMeans(is.na(training)) < 0.9]
training <- training[,-c(1:7)] 

nvz <- nearZeroVar(training)
training <- training[,-nvz]


inTrain  <- createDataPartition(training$classe, p=0.7, list=FALSE)
train <- training[inTrain, ]
test  <- training[-inTrain, ]
dim(train)
dim(test)

validation <- trainControl(method="cv", number=3, verboseIter=F)

#SVMLinear
modelFit1 <- train(classe~., data=train, method="svmLinear", trControl = validation, tuneLength = 5, verbose = F)
pred_model1 <- predict(modelFit1, test)
cm_mod1 <- confusionMatrix(pred_model1, factor(test$classe))
cm_mod1

#Random Forest
modelFit2 <- train(classe~., data=train, method="rf", trControl = validation, tuneLength = 5)
pred_model2 <- predict(modelFit2, test)
cm_mod2<- confusionMatrix(pred_model2, factor(test$classe))
cm_mod2

#rpart
modelFit3 <- train(classe~., data=train, method="rpart", trControl = validation, tuneLength = 5)
pred_model3 <- predict(modelFit3, test)
cm_mod3 <- confusionMatrix(pred_model3, factor(test$classe))
cm_mod3

#GBM
modelFit4 <- train(classe~., data=train, method="gbm", trControl = validation, tuneLength = 5, verbose = F)
pred_model4 <- predict(modelFit4, test)
cm_mod4 <- confusionMatrix(pred_model4, factor(test$classe))
cm_mod4


pred_randomForest <- predict(modelFit2, testing)
print(pred_randomForest)

pred_gbm <- predict(modelFit4,testing)
print(pred_gbm)
#Check
print(pred_randomForest == pred_gbm)
