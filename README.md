# DataScience-Classification
For this project, I first found the “mole2_dataset.csv” file. The data is about moles. It contains some features that will help me determine whether the mole is velus, melanoma, seborrheic keratosis or basal cell carcinoma. The last column is the response variable.Then I followed the steps below in the required order.<br>
**1.** Split your data into train (80%) and test (20%) datasets using caret’s createDataPartition function.<br>
**2.** Use the skim_to_wide function in skimr package, and provide descriptive stats for each column.<br>
**3.** Predict and impute the missing values with k-Nearest Neighbors using preProcess function in caret.<br>
**4.** After you impute missing values, use variable transformations. Convert all the numeric variables to range between 0 and 1, by setting method=range in preProcess function.<br>
**5.** Use caret’s featurePlot() function to visually examine how the predictors influence the predictor variable.<br>
**6.** <br>
**a.** Use train() function to build the machine learning model. Choose knn algorithm.<br>
**b.** Make predictions for test data using the predict() function.<br>
**c.** Construct the confusion matrix to compare the predictions (data) vs the actuals (reference). <br>
**7.** <br>
**a.** Use train() function to build the machine learning model. Choose random forest algorithm.<br>
**b.** Make predictions for test data using the predict() function.<br>
**c.** Construct the confusion matrix to compare the predictions (data) vs the actuals (reference). <br>
**8.** <br>
**a.** Use train() function to build the machine learning model. Choose naïve bayes classification algorithm.<br>
**b.** Make predictions for test data using the predict() function.<br>
**c.** Construct the confusion matrix to compare the predictions (data) vs the actuals (reference). <br>
**9.** Compare and make more and more comments about the final results you find in steps “6-8”.<br>

## The Project Summary
