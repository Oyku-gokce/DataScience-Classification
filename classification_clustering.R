library(caret)
mole <- mole2_dataset

mole$diagnosis <- factor(mole$diagnosis)
mole$sex <- factor(mole$sex)
mole$image_id <- factor(mole$image_id)

str(mole)

3)
preProcess_missingdata_model <- preProcess(mole, method='knnImpute')
preProcess_missingdata_model

I perform centering and scaling specifically to eliminate unit differences between numerical variables and to help the model perform better. I implemented missing value filling to solve the problem of missing data and include more data in the analysis. The fact that the ignored variables are not included in the analysis shows us that certain variables (3 variables) are not taken into account in the analysis.

install.packages("RANN")
# Use the imputation model to predict the values of missing data points
library(RANN)  # required for knnInpute
data_imputed <- predict(preProcess_missingdata_model, newdata = mole)
anyNA(data_imputed)

AnyNA returning false tells us that all missing values were completed successfully.

4)
preProcess_range_model <- preProcess(data_imputed, method='range', range=c(0,1))
data_transform <- predict(preProcess_range_model, data_imputed)

data_transform <- data_transform[, 3:14]

I continue our operations by removing the first 2 columns from our data that I think are not necessary for me.

1)
set.seed(123)
trainIndex <- createDataPartition(data_transform$diagnosis, p=0.8, list=FALSE)

trainData <- data_transform[ trainIndex,]
testData  <- data_transform[-trainIndex,]

nrow(trainData)
nrow(testData)

x=trainData[, 1:11]
y = trainData$diagnosis

2)
install.packages("skimr")
library(skimr)
skimmed <- skim_to_wide(trainData)
skimmed[,]

The values of the pixel_x variable seem to be generally concentrated around the mean. The fact that the standard deviation is low shows us that the values are generally distributed within the average range. It can be seen that the pixel_y variable does not contain missing data, the average value is 0.3662, and the values are distributed close to the average with low standard deviations.
5)
num_columns <- sapply(trainData, is.numeric)
num_data <- trainData[ , num_columns]

featurePlot(x = num_data,
            y = trainData$diagnosis, 
            plot = "boxplot")

The first thing that strikes us in box plots is that age_approx gives the impression of a normal distribution. I see that all the remaining variables have outliers values.There are many outliers for the corners attribute. When I look at the red attribute, I see that the data is highly dispersed for carcinoma and the least dispersed for keratosis. (The larger the box, the more dispersed it is, and the smaller it is, the less dispersed it is.)

For Carcinoma, the median and interquartile range values for each age group are similar, indicating that the data distribution for Carcinoma is relatively stable across age groups. For Melanoma, the data distribution for each age group is similar to Carcinoma, but with a slightly larger spread of values, suggesting that the data points for Melanoma are more spread out compared to Carcinoma. For Nevus, the median values for each age group are slightly higher compared to Carcinoma and Melanoma, while the interquartile range values are also slightly higher, indicating that Nevus data points are generally larger. For Keratosis, the median values for each age group are similar to Carcinoma and Melanoma, while the interquartile range values are slightly lower, indicating that Keratosis data points are generally smaller compared to Carcinoma and Melanoma.

6)
# Train the model using randomForest and predict on the training data itself.
model_knn = train(diagnosis ~ ., data=trainData, method='knn')


predictions <- predict(model_knn, newdata = testData)

# Compute the confusion matrix
confusionMatrix(reference = testData$diagnosis, data = predictions)

7)
install.packages("randomForest")
model_rf <- train(diagnosis ~ ., data = trainData, method = 'rf')

fitted <- predict(model_rf, newdata = testData)

confusionMatrix(reference = testData$diagnosis, data = fitted)

8)
install.packages("naivebayes")
model_nb <- train(diagnosis ~ ., data = trainData, method = 'naive_bayes')

predict <- predict(model_nb, newdata = testData)

confusionMatrix(reference = testData$diagnosis, data = predict)

