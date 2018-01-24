
## Authors: Ruchi Neema Gupta, Snehal Vartak
# Install library packages - e1071, ggplot2, reshape2, randomForest, rpart, rattle

library(e1071)  # for implementing Naives Bayes classifier
library(ggplot2) # for different plots
library(reshape2)
library (randomForest)  # for implementin random forest classifier
library(rpart)
library(rattle)


########################### testing and training data ####################################################################

trainFileName = "adult.data"; 
testFileName = "adult.test";

########### if the file does not exist then load it directly from web ##################################################

if (!file.exists (trainFileName))
  download.file (url = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                 destfile = trainFileName)

if (!file.exists (testFileName))
  download.file (url = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
                 destfile = testFileName)

colNames = c ("age", "workclass", "fnlwgt", "education", 
              "educationnum", "maritalstatus", "occupation",
              "relationship", "race", "sex", "capitalgain",
              "capitalloss", "hoursperweek", "nativecountry",
              "incomelevel")

#################################### read data in train ################################################################

train = read.table (trainFileName, header = FALSE, sep = ",",
                    strip.white = TRUE, col.names = colNames,na.strings = "?", stringsAsFactors = TRUE)

test = read.table (testFileName, header = FALSE, sep = ",",
                    strip.white = TRUE,skip = 1, col.names = colNames,na.strings = "?", stringsAsFactors = TRUE)


##################################################################################################################

str(train)
#Get an overiew of the data 
summary(train)
table (complete.cases (train))
# Summarize all data sets with NAs only
summary  (train [!complete.cases(train),])

# Distribution of the income level factor in the entire training data set.
table (train$incomelevel)

############################## MAKE A CLASS VARIABLE #################################################################
# train["class"]= ifelse(train["incomelevel"]== ">50K",1,0)

###################################### Cleaning data ###################################################################

index_train_complete= complete.cases(train) # returns a logical vector indicating which cases are complete
sapply(train, function(train) sum(is.na(train))) # The following function will tell the number of NA in each column
clean_train = na.omit(train)  # clean dataset;delete rows with NA data
test= na.omit(test)  # clean test if it contains NA


#clean_train = data.matrix(clean_train)
# convert dataframe to matrix

X = data.matrix(clean_train)


#######################################  DATA VISUALIZATION ##########################################################################

boxplot (age ~ incomelevel, data = clean_train, 
         main = "Age distribution for different income levels",
         xlab = "Income Levels", ylab = "Age", col = c("royalblue","orangered"))

boxplot (educationnum ~ incomelevel, data = clean_train, 
         main = "Highest Level of education distribution for different income levels",
         xlab = "Income Levels", ylab = "EducationNum", col = c("royalblue","orangered"))

boxplot (hoursperweek ~ incomelevel, data = clean_train, 
         main = "Hours per week distribution for different income levels",
         xlab = "Income Levels", ylab = "Hours per week", col = c("royalblue","orangered"))

boxplot (fnlwgt ~ incomelevel, data = clean_train, 
         main = "Final weight distribution for different income levels",
         xlab = "Income Levels", ylab = "Final weight", col = c("royalblue","orangered"))

#Plot density for each attribute using lattice
par(mfrow=c(3,5))
for(i in 1:15) {
  plot(density(X[,i]), main=colnames(X)[i])
}
par(mfrow=c(1,1))

ggplot(clean_train, aes(x = sex,fill=incomelevel))+ geom_bar(stat="count",position = "dodge")+
  ggtitle("Distribution of incomelevels w.r.t sex ") + scale_fill_brewer(palette = 'Set1')

# Skipped bar plots for educationnum, hoursperweek and age. Box plots are already done for these variables
#ggplot(clean_train, aes(x = educationnum,fill=incomelevel))+ geom_bar(stat="count",position = "dodge")+
 # scale_fill_brewer(palette = 'Set1')+ggtitle("Distribution of incomelevel w.r.t educationnum")

ggplot(clean_train, aes(x = workclass,fill=incomelevel))+ geom_bar(stat="count", position = "dodge")+
  scale_fill_brewer(palette = 'Set1')+ggtitle("Distribution of incomelevels w.r.t workclass")

#ggplot(clean_train, aes(x = occupation, fill=incomelevel))+ geom_bar(stat="count",position = "dodge")+
#  scale_fill_brewer(palette = 'Set1')+ggtitle("Distribution of incomelevels w.r.t occupation")

ggplot(clean_train, aes(x = maritalstatus, fill=incomelevel))+ geom_bar(stat="count",position = "dodge")+
  scale_fill_brewer(palette = 'Set1')+ ggtitle("Distribution of incomelevels wrt marital status")

ggplot(clean_train, aes(x = race, fill=incomelevel))+  geom_bar(stat="count",position = "dodge")+
  scale_fill_brewer(palette = 'Set1')+ggtitle("Distribution of incomelevels wrt race")

#ggplot(clean_train, aes(x = hoursperweek, fill=incomelevel))+ geom_bar(stat="count",position = "dodge",binwidth = 10)+
#  scale_fill_brewer(palette = 'Set1') + ggtitle("Distribution of incomelevels wrt hours per week")

#ggplot(clean_train, aes(x = age, fill=incomelevel))+ geom_bar(stat="count",position = "dodge",binwidth = 10)+
  #scale_fill_brewer(palette = 'Set1')+ggtitle("Distribution of incomelevels wrt age group")

#########################################  CLASSIFICATION USING NAIVE BAYES  #####################################################

# making model for naive bayes which will be used for prediction 
model_nb = naiveBayes(incomelevel ~.,data = clean_train)

# prediction on train and test data
prediction_nb_train = predict(model_nb,train[,-15])
prediction_nb_test = predict(model_nb,test[,-15])

# confusion matrix for train and test data
conf_mat_nb_train = table(prediction_nb_train, train[,15])
conf_mat_nb_test = table(prediction_nb_test,test[,15])

# print both confusion matrix
print("Confusion matrix for train data")
print(conf_mat_nb_train)
print("confusion matrix for test data")
print(conf_mat_nb_test)

# error rate for training and testing data
error_rate_test_nb =(conf_mat_nb_test[1,2]+conf_mat_nb_test[2,1])/(conf_mat_nb_test[1,2]+conf_mat_nb_test[2,1]+conf_mat_nb_test[1,1]+conf_mat_nb_test[2,2])
print("Error rate for Naive Bayes on test data")
print(error_rate_test_nb)

error_rate_train_nb =(conf_mat_nb_train[1,2]+conf_mat_nb_train[2,1])/(conf_mat_nb_train[1,2]+conf_mat_nb_train[2,1]+conf_mat_nb_train[1,1]+conf_mat_nb_train[2,2])
print("Error rate for Naive Bayes on train data")
print(error_rate_train_nb)


#######################################  CLASSIFICATION USING RANDOM FOREST  ###################################################

# genearte model for random forest
model_rf = randomForest(incomelevel ~ . , data = clean_train, mtry=2, ntree=1000,
                        keep.forest=TRUE, importance=TRUE)

levels(test$nativecountry) <- levels(clean_train$nativecountry)

# predict on train and test data using above generated model
prediction_rf_test = predict(model_rf, test[,-15])
prediction_rf_train = predict(model_rf, clean_train[,-15])

# print the confusion matrix for both test and train data
conf_mat_rf_train = table(prediction_rf_train,clean_train[,15])
conf_mat_rf_test = table(prediction_rf_test,test[,15])

# print both confusion matrix
print("Confusion matrix for train data")
print(conf_mat_rf_train)
print("confusion matrix for test data")
print(conf_mat_rf_test)

# print error rate for both train and test data
error_rate_test_rf =(conf_mat_rf_test[1,2]+conf_mat_rf_test[2,1])/(conf_mat_rf_test[1,2]+conf_mat_rf_test[2,1]+conf_mat_rf_test[1,1]+conf_mat_rf_test[2,2])
print("Error rate for Random forest on test data")
print(error_rate_test_rf)

error_rate_train_rf =(conf_mat_rf_train[1,2]+conf_mat_rf_train[2,1])/(conf_mat_rf_train[1,2]+conf_mat_rf_train[2,1]+conf_mat_rf_train[1,1]+conf_mat_rf_train[2,2])
print("Error rate for Naive Bayes on train data")
print(error_rate_train_rf)


##############################  Classification using decison trees ##############################################################

# model generation for decision trees
tree  = rpart(incomelevel ~ ., data = clean_train, method ="class")
fancyRpartPlot(tree, main = "Decision tree for Adult dataset")


# predictions on test and train data
prediction_dt_test=  predict(tree, test[,-15])
prediction_dt_train=  predict(tree, clean_train[,-15])
predictions_test = ifelse(prediction_dt_test[,1] >= .5, " <=50K", " >50K")
predictions_train = ifelse(prediction_dt_train[,1] >= .5, " <=50K", " >50K")


# print the confusion matrix for both test and train data
conf_mat_dt_train = table(predictions_train,clean_train[,15])
conf_mat_dt_test = table(predictions_test,test[,15])

# print both confusion matrix
print("Confusion matrix for train data")
print(conf_mat_dt_train)
print("confusion matrix for test data")
print(conf_mat_dt_test)



# print error rate for both train and test data
error_rate_test_dt =(conf_mat_dt_test[1,2]+conf_mat_dt_test[2,1])/(conf_mat_dt_test[1,2]+conf_mat_dt_test[2,1]+conf_mat_dt_test[1,1]+conf_mat_dt_test[2,2])
print("Error rate for Random forest on test data")
print(error_rate_test_dt)

error_rate_train_dt =(conf_mat_dt_train[1,2]+conf_mat_dt_train[2,1])/(conf_mat_dt_train[1,2]+conf_mat_dt_train[2,1]+conf_mat_dt_train[1,1]+conf_mat_dt_train[2,2])
print("Error rate for Naive Bayes on train data")
print(error_rate_train_dt)

# accuracy <- round(sum(predictions == test[,15])/length(predictions), digits = 4)
# print(paste("The model correctly predicted the test outcome ", accuracy*100, "% of the time", sep=""))
# predictions <- ifelse(outcomes[,1] >= .5, " <=50K", " >50K") 
# plot(fit, uniform=TRUE,main="Classification Tree without pruning")
# text(fit, use.n=TRUE, all=TRUE, cex=.8)
# 
# 
# #pruning the tree
# pfit= prune(fit, cp=0.013441)
# 
# # plot the pruned tree 
# plot(pfit, uniform=TRUE, main="Pruned Classification Tree for g3 using class as method")
# text(pfit, use.n=TRUE, all=TRUE, cex=.8)
