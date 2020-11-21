
# load packages
library("tidyverse")
library("caret")
library("mlr")
library("pROC")

# read in the data
cc = read_csv("data/cc-sub.csv")

# feature engineering
cc$Class = factor(as.numeric(factor(cc$Class))) # 1 -> fraud, 2 -> genuine

# function to determine proportion of NAs in a vector
na_prop = function(x) {
  mean(is.na(x))
}

# check proportion of NAs in each column
print(sapply(cc, na_prop)) # no NAs

# get ratio of classes
classes = table(cc$Class)
print(classes) # accuracy is a bad metric; need to balance data

# make a classification task
task = makeClassifTask(data = cc, target = "Class")

###### OVERSAMPLE ######
print("OVERSAMPLE TEST")

# oversample data [fraud]
oversampling_rate = ceiling(classes[2] / classes[1])
oversample_data = getTaskData(smote(task, oversampling_rate, nn = 10))
print(table(oversample_data$Class))

# split into train-test
set.seed(42)
trn_idx = createDataPartition(oversample_data$Class, p = 0.80, list = TRUE)
trn = oversample_data[trn_idx$Resample1, ]
tst = oversample_data[-trn_idx$Resample1, ]

# train models
cv_5 = trainControl(method = "cv", number = 5)
tree_mod = caret::train(form = Class ~ ., data = trn, method = "rpart", trControl = cv_5)
print(tree_mod)
knn_mod = caret::train(form = Class ~ ., data = trn, method = "knn", trControl = cv_5)
print(knn_mod)
gbm_mod = caret::train(form = Class ~ ., data = trn, method = "gbm", trControl = cv_5, verbose = FALSE)
print(gbm_mod)

# calculate test accuracy
print("ACCURACIES:")
tree_preds = predict(tree_mod, newdata = tst, type = "raw")
print(mean(tree_preds == tst$Class)) # tree

knn_preds = predict(knn_mod, newdata = tst, type = "raw")
print(mean(knn_preds == tst$Class)) # knn

gbm_preds = predict(gbm_mod, newdata = tst, type = "raw")
print(mean(gbm_preds == tst$Class)) # gbm

# calculate test AUC
print("AUCs:")
print(auc(as.numeric(tst$Class), as.numeric(tree_preds))) # tree
print(auc(as.numeric(tst$Class), as.numeric(knn_preds))) # knn
print(auc(as.numeric(tst$Class), as.numeric(gbm_preds))) # gbm

# print confusion matrix of best model
confusionMatrix(gbm_preds, tst$Class)
