#### Titanic Survival Prediction

*My* Approach

------------------------------------------------

### Libraries used

'''sh
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(neuralnet)
library(e1071)
library(caret)
library(randomForest)
library(party)
'''

### Getting the Data Into R

'''sh
train<-read.csv('train.csv')
test<- read.csv('test.csv')
'''

### Combining the data so as to manipulate and set to right format

'''sh
test$Survived<-NA
combi<-rbind(train,test)
'''

### Extracting the information from Names to predict missing ages and some other manipulations
'''sh
combi$Name <- as.character(combi$Name)
strsplit(combi$Name[1], split='[,.]')
strsplit(combi$Name[1], split='[,.]')[[1]]
strsplit(combi$Name[1], split='[,.]')[[1]][2]
combi$Title <- sapply(combi$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
combi$Title <- sub(' ', '', combi$Title)
combi$Title[combi$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Mr'
combi$Title[combi$Title %in% c('Col','Dr','Jonkheer','Rev')] <- 'Mr'
combi$Title[combi$Title %in% c('the Countess','Mme','Dona','Lady')] <- 'Mrs'
combi$Title[combi$Title %in% c('Mlle')] <- 'Miss'
combi$Fare[is.na(combi$Fare)] <- mean(combi$Fare, na.rm=TRUE)
combi$FamId<- combi$SibSp+ combi$Parch 
'''
### Predicting NAs of Age
I am aware that loops are slow in R, but since I am new to R and the data being small I went with it

'''sh
for (i in 1:1309) {
  if(is.na(combi$Age[i])){
    if(combi$Title[i]=="Master"){combi$Age[i]<-sample(1:17,1)}
    else if(combi$Title[i]=="Mr"){combi$Age[i]<-sample(18:50,1)}
    else if(combi$Title[i]=="Miss"){combi$Age[i]<-sample(18:29,1)}
    else if(combi$Title[i]=="Mrs"){combi$Age[i]<-sample(30:60,1)}
    else{combi$Age[i]<-sample(18:60,1)}
  }
}
'''

### Predicting the errors in Fares
'''sh
for(i in 1:1309){
  if(combi$Fare[i]<7 && combi$Pclass[i]==1){combi$Fare[i]<-50}
  else if(combi$Fare[i]<7 && combi$Pclass[i]==2){combi$Fare[i]<-13}
  else if(combi$Fare[i]<7 && combi$Pclass[i]==3){combi$Fare[i]<-7}
  else{combi$Fare[i]<-combi$Fare[i]}
}
combi$Fare[is.na(combi$Fare)] <- mean(combi$Fare, na.rm=TRUE)
'''
### Splitting back the train and test data and factoring
'''sh
train <- combi[1:891,]
test <- combi[892:1309,]
factor(train$Sex, c("male", "female"), labels = c(1, 0))
factor(test$Sex, c("male", "female"), labels = c(1, 0))
factor(train$Embarked, c("C", "Q","S"), labels = c(1, 2,3))
factor(test$Embarked, c("C", "Q","S"), labels = c(1, 2,3))
'''

### Model Fitting

**Decision Trees**
'''sh
fit <- rpart(Survived ~ Sex + Age + FamId + Pclass + Fare , data = train, method="class")
fancyRpartPlot(fit)
final_result4 <- predict(fit, test, type = "class")
'''

**Neural Network**
'''sh
m <- model.matrix( ~ Survived + Pclass + Sex + Age + SibSp + Parch + Fare, data = train)
net <- neuralnet(Survived ~ Sexmale + Age + Pclass + SibSp + Parch, data=m, hidden = 10, threshold = 0.1)
plot(net)

test_temp<-subset(test,select=c("Pclass","Sex","Age","SibSp","Parch","Fare"))
factor(test_temp$Sex, c("male", "female"), labels = c(1, 0))
n <- model.matrix( ~ Pclass + Sex + Age + SibSp + Parch + Fare, data = test_temp)
prediction<-compute(net, n[,2:6])

for(i in 1:length(prediction$net.result)){
  if(prediction$net.result[i]>0.6){prediction$net.result[i]<-1}
  else{prediction$net.result[i]<-0}
}
'''

**SVM Modelling**
'''sh
train_svm<-train[,c("Age","Sex","Pclass","SibSp","Parch","Survived")]
svm.model<-svm(Survived ~ . , data = train_svm, kernel="radial")

test_svm<-test[,c("Age","Sex","Pclass","SibSp","Parch")]
preds<-predict(svm.model, test_svm)

for(i in 1:length(preds)){
  if(preds[i]>0.5){preds[i]<-1}
  else{preds[i]<-0}
}
'''
**Random Forests**
'''sh
set.seed(415)
fit2<- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + Fare + Embarked + FamId + Parch + SibSp, data=train, importance=TRUE, ntree=2000)
varImpPlot(fit)
Pred_rf <- predict(fit2, test)
'''

**C Forest**
'''sh
set.seed(415)
fit <- cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + FamId, data = train, controls=cforest_unbiased(ntree=2000, mtry=3))
Pred_cf <- predict(fit, test, OOB=TRUE, type = "response")
'''
There are many more models to fit this discrete variables. But, on research I came to know that the accuracy doesn't go beyond 82%. Since I scored a 80.3% accuracy
I am quite satisfied with it. This analysis was just a training excercise to get started with R and Machine Learning and it has been a great one.

Just to give the information: Decision trees scored the highest accuracy with 80.3% and with SVM, Random forests  & C forests the accuracy ranged in between 76% to 79%.
Neural Networks performed with a 39% accuracy (I am not surprised) and thus giving the lowest perfomance. Of course the perfomance of the neural networks can be
increased by feature engineering and several other paramenters, but since I was just experimenting I would dwell into neural networks with other projects. 

