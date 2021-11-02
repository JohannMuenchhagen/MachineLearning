# Final excersise

This project will use some trainingdata. The goal is to predict the manner in wich they did there excercises.
Since this is a classification problem, the following tree algorithms were chosen for this purpose.
- SVMLinear
- Random Forest
- rpart
- GBM

## Step 1: Download the data as csv file.
## Step 2: Load the required packages
```
library(lattice)
library(caret)
library(kernlab)
library(rattle)
```
## Step 3: Read the data and set seed
First I uploaded the csvs into the sandbox. Next, I read in the CSV files using the read.csv() command. Last I set the seed to 12345 for the purpose of repeatability.
```
training <- read.csv("~/pml-training.csv")
testing <- read.csv("~/pml-testing.csv")

set.seed(12345)
```
## Step 4: Clean the data
First, the proportion of NA lines was reduced. Now the variance was increased with nearZeroVar().
```
training <- training[,colMeans(is.na(training)) < 0.9]
training <- training[,-c(1:7)] 

nvz <- nearZeroVar(training)
training <- training[,-nvz]
```

## Step 5: Building a training and testing dataset
With createDataPartition() the data set was divided into a training set and a test set where 70% of the data is used for training.
Now the data was loaded into separate array. Last the validation() method is initialized.
```
inTrain  <- createDataPartition(training$classe, p=0.7, list=FALSE)
train <- training[inTrain, ]
test  <- training[-inTrain, ]
dim(train)
dim(test)

validation <- trainControl(method="cv", number=3, verboseIter=F)
```
## Step 6: Train and test the SVMLinear model
To train and test the model I use the code below.
```
modelFit1 <- train(classe~., data=train, method="svmLinear", trControl = validation, tuneLength = 5, verbose = F)
pred_model1 <- predict(modelFit1, test)
cm_mod1 <- confusionMatrix(pred_model1, factor(test$classe))
cm_mod1
```
The result of the model was as follows.
```
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1566  159   89   70   64
         B   26  815  111   43  146
         C   36   63  778  100   60
         D   37   20   28  718   61
         E    9   82   20   33  751

Overall Statistics
                                          
               Accuracy : 0.7864          
                 95% CI : (0.7757, 0.7968)
    No Information Rate : 0.2845          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.7281          
                                          
 Mcnemar's Test P-Value : < 2.2e-16       

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9355   0.7155   0.7583   0.7448   0.6941
Specificity            0.9093   0.9313   0.9467   0.9703   0.9700
Pos Pred Value         0.8039   0.7143   0.7502   0.8310   0.8391
Neg Pred Value         0.9726   0.9317   0.9488   0.9510   0.9337
Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
Detection Rate         0.2661   0.1385   0.1322   0.1220   0.1276
Detection Prevalence   0.3310   0.1939   0.1762   0.1468   0.1521
Balanced Accuracy      0.9224   0.8234   0.8525   0.8576   0.8321
```
An accuracy of 78.64% is quite good, so lets see if the other models are more accurate.
## Step 7: Train and test the Random Forest model
To train and test the model I use the code below.
```
modelFit2 <- train(classe~., data=train, method="rf", trControl = validation, tuneLength = 5)
pred_model2 <- predict(modelFit2, test)
cm_mod2<- confusionMatrix(pred_model2, factor(test$classe))
cm_mod2
```
The result was as follows.
```
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1673    4    0    0    0
         B    1 1133    3    0    0
         C    0    2 1021    9    0
         D    0    0    2  954    1
         E    0    0    0    1 1081

Overall Statistics
                                          
               Accuracy : 0.9961          
                 95% CI : (0.9941, 0.9975)
    No Information Rate : 0.2845          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.9951          
                                          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9994   0.9947   0.9951   0.9896   0.9991
Specificity            0.9991   0.9992   0.9977   0.9994   0.9998
Pos Pred Value         0.9976   0.9965   0.9893   0.9969   0.9991
Neg Pred Value         0.9998   0.9987   0.9990   0.9980   0.9998
Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
Detection Rate         0.2843   0.1925   0.1735   0.1621   0.1837
Detection Prevalence   0.2850   0.1932   0.1754   0.1626   0.1839
Balanced Accuracy      0.9992   0.9969   0.9964   0.9945   0.9994
```
The accuracy of 99.61% is close to perfect. 
## Step 8: Train and test the rpart model
To train and test the model I use the code below.
```
modelFit3 <- train(classe~., data=train, method="rpart", trControl = validation, tuneLength = 5)
pred_model3 <- predict(modelFit3, test)
cm_mod3 <- confusionMatrix(pred_model3, factor(test$classe))
cm_mod3
```
The result was as follows.
```
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1525  484  499  423  153
         B   29  351   36    9  138
         C   77  124  423  126  143
         D   39  180   68  406  167
         E    4    0    0    0  481

Overall Statistics
                                          
               Accuracy : 0.5414          
                 95% CI : (0.5285, 0.5542)
    No Information Rate : 0.2845          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.402           
                                          
 Mcnemar's Test P-Value : < 2.2e-16       

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9110  0.30817  0.41228  0.42116  0.44455
Specificity            0.6298  0.95533  0.90327  0.90774  0.99917
Pos Pred Value         0.4945  0.62345  0.47368  0.47209  0.99175
Neg Pred Value         0.9468  0.85194  0.87921  0.88896  0.88870
Prevalence             0.2845  0.19354  0.17434  0.16381  0.18386
Detection Rate         0.2591  0.05964  0.07188  0.06899  0.08173
Detection Prevalence   0.5240  0.09567  0.15174  0.14613  0.08241
Balanced Accuracy      0.7704  0.63175  0.65778  0.66445  0.72186
```
The accuracy of 54.14% is a bit better than guessing.
## Step 9: Train and test the GBM model
To train and test the model I use the code below.
```
modelFit4 <- train(classe~., data=train, method="gbm", trControl = validation, tuneLength = 5, verbose = F)
pred_model4 <- predict(modelFit4, test)
cm_mod4 <- confusionMatrix(pred_model4, factor(test$classe))
cm_mod4
```
The result as follows.
```
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1667    6    0    0    0
         B    7 1125    7    0    0
         C    0    8 1017   11    1
         D    0    0    2  953    3
         E    0    0    0    0 1078

Overall Statistics
                                          
               Accuracy : 0.9924          
                 95% CI : (0.9898, 0.9944)
    No Information Rate : 0.2845          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.9903          
                                          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: A Class: B Class: C
Sensitivity            0.9958   0.9877   0.9912
Specificity            0.9986   0.9971   0.9959
Pos Pred Value         0.9964   0.9877   0.9807
Neg Pred Value         0.9983   0.9971   0.9981
Prevalence             0.2845   0.1935   0.1743
Detection Rate         0.2833   0.1912   0.1728
Detection Prevalence   0.2843   0.1935   0.1762
Balanced Accuracy      0.9972   0.9924   0.9936
                     Class: D Class: E
Sensitivity            0.9886   0.9963
Specificity            0.9990   1.0000
Pos Pred Value         0.9948   1.0000
Neg Pred Value         0.9978   0.9992
Prevalence             0.1638   0.1839
Detection Rate         0.1619   0.1832
Detection Prevalence   0.1628   0.1832
Balanced Accuracy      0.9938   0.9982
```
The accuracy of 99.24% is close to the accuracy of the random forest model.
## Step 10: Predict 
Since the Random Forest model and the GBM model have nearly identical accuracy, both models are tested separately. Perhaps the small deviation provides a different result.
### 1. Start with the Random Forest model.
To predict with these model I use the predict() method.
```
pred_randomForest <- predict(modelFit2, testing)
print(pred_randomForest)

```

The predict values are:
```
[1] B A B A A E D B A A B C B A E E A B B B
Levels: A B C D E
```
### 2. Predicton with th GBM model
To predict with these model I use the predict() method.
```
pred_gbm <- predict(modelFit4,testing)
print(pred_gbm)
```
The predicted values are:

```
[1] B A B A A E D B A A B C B A E E A B B B
Levels: A B C D E
```
### 3. Lets check if the results are identical
To check these I use a simple boolean check in the print() method.
```
print(pred_randomForest == pred_gbm)
```
The result is:
```
[1] TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE
[10] TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE
[19] TRUE TRUE
```
As you can see the results are identical and therefore it doesn't matter which model you choose.
