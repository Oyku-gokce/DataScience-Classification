1)
library(caret)
mole <- mole2_dataset
mole$sex[mole$sex == "male"] <- "1"
mole$sex[mole$sex == "female"] <- "0"

str(mole)

# See top 6 rows and 10 columns
head(mole[, 1:14])
ncol(mole)

set.seed(123)
trainIndex <- createDataPartition(mole$diagnosis, p=0.8, list=FALSE)

trainData <- mole[ trainIndex,]
testData  <- mole[-trainIndex,]

nrow(trainData)
nrow(testData)

x=trainData[, 1:13]
y = trainData$diagnosis


2)
install.packages("skimr")
library(skimr)
skimmed <- skim_to_wide(trainData)
skimmed[,]

3)
preProcess_missingdata_model <- preProcess(trainData, method='knnImpute')
preProcess_missingdata_model

That is, it has centered (subtract by mean) 11 variables, ignored 3, used k=5 (considered 5 nearest neighbors) to predict the missing values and finally scaled (divide by standard deviation) 11 variables.

install.packages("RANN")
# Use the imputation model to predict the values of missing data points
library(RANN)  # required for knnInpute
trainData <- predict(preProcess_missingdata_model, newdata = trainData)
anyNA(trainData)

All the missing values are successfully imputed.

# One-Hot Encoding
# Creating dummy variables is converting a categorical variable to as many binary variables as here are categories.
dummies_model <- dummyVars(sex~. , data=trainData)

# Create the dummy variables using predict. The Y variable (Purchase) will not be present in trainData_mat.
trainData_mat <- predict(dummies_model, newdata = trainData)

# # Convert to dataframe
trainData <- data.frame(trainData_mat)

# # See the structure of the new dataset
str(trainData)

4)
preProcess_range_model <- preProcess(trainData, method='range')
trainData <- predict(preProcess_range_model, newdata = trainData)

# Append the Y variable
trainData$diagnosis <- y

apply(trainData[,c(1,3:4,6:13)], 2, FUN=function(x){c('min'=min(x), 'max'=max(x))})

trainData$diagnosis <- factor(trainData$diagnosis)

5)
featurePlot(x = trainData[,c(5:6,8:13)],
            y = trainData$diagnosis, 
            plot = "density",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")))

6)
testData$diagnosis <- factor(testData$diagnosis)

a)
# Train the model using randomForest and predict on the training data itself.
model_knn = train(diagnosis ~ .- X - image_id - pixels_x - pixels_y - full_img_size, data=trainData, method='knn')

b)
testData$image_id <- factor(testData$image_id, levels = levels(trainData$image_id))
predictions <- predict(model_knn, newdata = testData)
head(predictions)

# Step 1: Impute missing values 
testData2 <- predict(preProcess_missingdata_model, testData)
# Step 2: Create one-hot encodings (dummy variables)

# Step 3: Transform the features to range between 0 and 1
testData3 <- predict(preProcess_range_model, testData2)

# View
head(testData3[, 1:10])

# Predict on testData
predicted <- predict(model_knn, testData3)
head(predicted)

predicted <- predict(model_knn, testData3)
head(predicted)

c)
# Compute the confusion matrix
confusionMatrix(reference = testData$diagnosis, data = predicted, mode='everything')

7)
# Train the model using randomForest and predict on the training data itself.
install.packages("randomForest")
model_rf <- train(X ~ ., data = trainData, method = 'rf')
fitted <- predict(model_rf)

# trainData veri setini içeren formula ifadesini kontrol et
formula <- X ~ .
print(formula)

# Eğitim veri setindeki değişken adlarını kontrol et
print(names(trainData))

# Tahmin yapılacak yeni veri setindeki değişken adlarını kontrol et
# newData, tahmin yapılacak yeni veri setini temsil eder

# Random Forest modelini eğitmek için train fonksiyonunu kullanma
model_rf <- train(formula, data = trainData, method = 'rf')

# Eğitilmiş modeli kullanarak tahminler yapma
fitted <- predict(model_rf, newdata = newData)

