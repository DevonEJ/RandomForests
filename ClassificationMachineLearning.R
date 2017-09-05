# This data was obtained from https://www.kaggle.com/c/titanic 

# 1. Set up ##########################################################################
# Set working directory
setwd("C:/Projects/Machine Learning")

# Load Titanic survival training & test datasets
# Dataset was split into test and training automatically by Kaggle.com 
training <- read.csv("train.csv")
testing <- read.csv("test.csv")

# Load packages
library(rpart)
library(rpart.plot)

# Take a look at the two datasets
glimpse(training)
glimpse(testing)

## Are there any more features we could draw out from these data?? 

# 2. Model building using training data ##################################################################
# Build decision tree classification model to model survival
survival_model <- rpart(Survived ~ Age + Pclass + Sex, training, method = "class", control = rpart.control(cp = 0))
  
# View model and plot visual representation
rpart.plot(survival_model, type = 3, box.palette = c("red", "green"), fallen.leaves = TRUE)

# Post-pruning the decision tree 
# First determine the optimal complexity parameter using the cp plot to visualise
plotcp(survival_model)

# Now use that 0.0083 value to prune the model back and reduce complexity
survival_model2 <- prune(survival_model, cp = 0.0083)

# View model and plot visual representation again to see how it's changed
rpart.plot(survival_model2, type = 3, box.palette = c("red", "green"), fallen.leaves = TRUE)

# 2. Predicting survival using test data ################################################################## 
# Predict survival using original model
pred <- predict(survival_model, testing, type = "class")

# Predict survival using simplified (pruned) tree model
pred2 <- predict(survival_model2, testing, type = "class")

# Performance assessment - create confusion matrix to compare predicted and actual survival
##table(loans_test$pred, loans_test$outcome)

# Compute the accuracy of testing dataset - gives a % accuracy score
##mean(loans_test$pred == loans_test$outcome)  

#################################################################################################
# Applying Random Forests algorithmn to the problem as ensemble method
# Load packages
library(randomForest)

# Build forest
#### object <- randomForest(outcome_var ~ var2 + var3, data = data, ntree = 500, mtry = sqrt(p))
# use predict and performance check as normal
