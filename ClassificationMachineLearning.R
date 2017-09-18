# These data were obtained from https://www.kaggle.com/c/titanic 
# This script is part of a Kaggle competition submission 

# Clear R environment 
rm(list=ls())

#################################################################################################
# 1. Set up
#################################################################################################
# Set working directory
setwd("C:/Projects/MachineLearning")

# Load packages
library(rpart) # decision tree model
library(rpart.plot) # plotting decision tree model
library(mice) # Missing value imputation - prediction
library(dplyr) # Data manipulation/cleaning
library(randomForest) # randomForest model
library(ggplot2) # Visualisation 
library(scales) # Visualisation
library(rattle) # tree model visualisation 
library(party) # conditonal inference forest model

# Load Titanic survival training & test datasets
training <- read.csv("train.csv")
testing <- read.csv("test.csv")
full_data <- bind_rows(training, testing)

# Take a look at the data
str(full_data)
glimpse(full_data)

#################################################################################################
# 2. Feature engineering - Are there any more features we could draw out 
#################################################################################################
###############################
# A. Titles - Let's extract the titles from 'Name' into a new column 'Title' using regex
full_data$Title <- gsub('(.*, )|(\\..*)', "", full_data$Name) 

# Let's see how many of each different title we have in the dataset
count_test <- group_by(full_data, Title) %>%
  count(Title)

# Create a vector of the rarest (=< 8 count) titles
rare_titles <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')

# Reassign the French and rare titles to match more common titles
full_data$Title <- gsub("Mlle", "Miss", full_data$Title)
full_data$Title <- gsub("Ms", "Miss", full_data$Title)
full_data$Title <- gsub("Mme", "Mrs", full_data$Title)

full_data$Title[full_data$Title %in% rare_titles] <- 'Rare'

unique(full_data$Title) # We now have 5 unique titles to work with!

###############################
# B. Surname - helps us infer families 
# Split surnames out of the Name column, similar to as with Titles above 
full_data$Surname <- sapply(full_data$Name, 
                            function(x) strsplit(x, split = "[,.]")[[1]][1])

unique(full_data$Surname) # We now have 875 unique surnames to work with!

##############################
# C. Families - let's create a family size variable based on the 'Sibsp' (# siblings aboard)
# and 'Parch' (# parents/children aboard)
full_data <- as_tibble(full_data) %>% 
  mutate(FamSize = SibSp + Parch + 1) # '+ 1' to include the person themselves

# Let's add a variable indicating each person's family group
full_data$Family <- paste(full_data$Surname, full_data$FamSize, sep='_')

# Is there actually a relationship between newly created 'FamSize' and 'Survived'?
full_data %>%
  filter(Survived != "NA") %>%
  ggplot(aes(x = FamSize, fill = factor(Survived))) +
  geom_bar(stat='count', position='dodge') +
  scale_x_continuous(breaks=c(1:11)) +
  labs(x = 'Family Size') 

# Plot shows that FamSize of 1 or > 4 means less chance of surviving here
# Let's capture that in a discretized FamSize variable ('FamSizeGroup') to make life easier
full_data$FamSizeGroup[full_data$FamSize == 1] <- 'single'
full_data$FamSizeGroup[full_data$FamSize < 5 & full_data$FamSize > 1] <- 'small'
full_data$FamSizeGroup[full_data$FamSize > 4] <- 'large'
# We now have 3 family grouping levels to work with! - single, small and large families 

# Let's see if discretizing the 'FamSize' variable has preserved the 'Survived ~ FamSize' relationship
mosaicplot(table(full_data$FamSizeGroup, full_data$Survived), main = 'Family Size by Survival', shade = TRUE)
# Yep - It's still better to be either alone or in a small family!

##################################
# D. Cabin information - we can split out where passengers slept on the boat by deck level
#Take a look at the 'Cabin' variable first - there are a lot of missing values here so we can't do too much
full_data$Cabin

# Let's split out the letter at start of 'Cabin' references first to create 'DeckRef' variable...
# ...this appears to refer to the cabin deck with the number following it being the cabin number
full_data$DeckRef <- factor(sapply(full_data$Cabin, 
                                   function(x) strsplit(x, NULL)[[1]][1]))

##################################################################################################
# 3. Missing value imputations - cleaning up the dataset
# A & B - Inference method
# C- Predictive method (using 'Multivariate Imputation by Chained Equations' (MICE))
##################################################################################################
#################################
# Let's get an overall picture of completeness of the data
md.pattern(full_data) # Output shows we have 133 complete rows, lots of missing values 
summary(full_data) # 1014 NA values in dataset

###############################
# A. Missing Embark Point 
summary(full_data$Embarked) # 2 missing Embarked values 
which(full_data$Embarked == '') # Passengers 62 & 830 are missing Embarked values
embark_missing <- full_data %>%
  filter(Embarked == "")

# Let's use the price paid and passenger class variables to infer what embark location might have been

# First, remove these 2 passengers from dataset to leave only passengers we have complete data for
'%ni%' <- Negate('%in%') # Reverse function of the %in% operand, in prep for filtering below
passenger_ids <- c(62, 830)
embark_data <- full_data %>%
  filter(PassengerId %ni% passenger_ids)

# Test that they have been removed - we should see those two passengers only within 'test' dataset
test <- anti_join(full_data, embark_data)

# Let's check that there's actually a relationship between PClass, Fare & Embark visually first
ggplot(embark_data, aes(x = Embarked, y = Fare, fill = factor(Pclass))) +
  geom_boxplot() +
  geom_hline(aes(yintercept = 80), # Draw a red line to show the $80 Fare passengers 62 & 830 paid
             colour='red', lwd = 2) +
  scale_y_continuous(labels = dollar_format())

# An $80 Fare is just above the median paid for passengers leaving from embarkation point 'C'
# So we can infer that these passengers embarked from C, and replace their 'NA's with 'C's
full_data[passenger_ids, "Embarked"] <- 'C'

#Test that these passengers''Embarked' was assigned correctly
full_data[passenger_ids, "Embarked"]

###############################
# B. Missing Fare  
summary(full_data$Fare) # 1 missing fare
which(is.na(full_data$Fare)) # Passenger 1044 
full_data[1044,]

# Remove this passenger from the dataset
fare_data <- full_data[-1044,]

# We can visualise those passengers with the same Pclass and embarkation point to work out a sensible Fare estimate
ggplot(fare_data[fare_data$Embarked == "S" & fare_data$Pclass == "3", ], aes(x = Fare)) +
  geom_density() +
  geom_vline(aes(xintercept = median(Fare, na.rm = TRUE)), # Add red line at the median fare value
             colour = 'red', lwd = 1) +
  scale_x_continuous(labels = dollar_format()) 

# Let's assign that median fare value to passenger 1044
full_data$Fare[1044] <- median(fare_data[fare_data$Embarked == "S" & fare_data$Pclass == "3",]$Fare)
full_data[1044,] # We infer that they paid $8.05 

#####################
# C. Missing Ages - There are a lot of these, so we will use a predictive method (MICE) to predict replacements
# MICE will predict 'plausible' values for Age based on other columns of data fed to the model
sum(is.na(full_data$Age)) # 263 missing ages

# First, factorise all of the factor variables in the dataset
# Create vector, then factorising function to apply to it
factors <- c('PassengerId','Pclass','Sex','Embarked',
                 'Title','Surname','Family','FamSize')

full_data[factors] <- lapply(full_data[factors], function(x) as.factor(x))

# MICE imputation using 'rf' (randomForest imputation) - multiplied over 5 datasets ('m = 5')
# MICE uses other collumns to predict Age column, so we remove columns which aren't relevant to that prediction, eg. Name
mice_model <- mice(full_data[, !names(full_data) %in% c('PassengerId','Name','Ticket','Cabin','Family','Surname','Survived')], m = 5, method = 'rf', seed = 123)

#Let's have a look at the imputations
# This shows the imputed values for the Age variable, calculated by each of the 5 datasets the function generated
# You can see that passenger 6 has been calculated as having ages of 28.00, 32.00, 39.00, 21.0 & 39.0
mice_model$imp$Age # Do any of these look implausible? (Eg. datasets 3 & 5 have a lot of extreme age estimates)

# Now, use let's create a full dataset using these imputed Age values - I'm going to choose dataset 2 to complete with, it looks sensible
complete_data <- complete(mice_model, 2)

# Let's check that worked - are their any NAs left in Age?
sum(is.na(complete_data$Age)) # Nope!

# Let's compare the original Age distribution to the imputed one to check for implausible values
par(mfrow = c(1,2))
hist(full_data$Age, freq = F, main = 'Original Age Dist.', 
     col ='darkgreen', ylim = c(0,0.04))
hist(complete_data$Age, freq = F, main = 'MICE Age Dist.', 
     col ='lightgreen', ylim = c(0,0.04)) 

# There aren't any weird outliers, so let's plug the imputed Age values into our original dataset
full_data$Age <- complete_data$Age

# Check that worked
sum(is.na(full_data$Age)) # Yep!

##########################
# Finally, let's check the completeness of the data
md.pattern(full_data)

#############################################################################################
# 4. A bit more feature engineering...
############################################################################################
#####################################
# Now that we have complete Age data, after missing value imputation, we can make Age easier to work with
# A. Age -  Let's create adult & child groupings under AgeGroup to reduce the complexity of individual ages
children <- full_data %>%
  filter(Age <= 17) %>%
  mutate(AgeGroup = "child")

adults <- full_data %>%
  filter(Age >= 18) %>%
  mutate(AgeGroup = "adult")

full_data <- full_join(children, adults)
nrow(full_data)

# Are these new AgeGroup categories useful in predicting Survived along with Sex?
# This shows number of survivors in each sub-grouping
aggregate(Survived ~ AgeGroup + Sex, data = full_data, FUN = sum)

# This shows total number of people in each sub-grouping
aggregate(Survived ~ AgeGroup + Sex, data = full_data, FUN = length)

# And this will show proportion of survival for each sub-grouping - this does seem to be a useful feature
aggregate(Survived ~ AgeGroup + Sex, data = full_data, FUN = function(x) {sum(x)/length(x) * 100})

##########################
# B. Fare - let's bin the Fare variable to make this easier to use too 
# We'll use 3 bins: $0-20, $20 - 30, and $30+

cheap <- full_data %>%
  filter(Fare <= 20) %>%
  mutate(FareGroup = "cheap")

medium <- full_data %>%
  filter(Fare > 20 & Fare <= 30) %>%
  mutate(FareGroup = "medium")

expensive <- full_data %>%
  filter(Fare > 30) %>%
  mutate(FareGroup = "expensive")

full_data <- full_join(cheap, medium)
full_data <- full_join(full_data, expensive)
nrow(full_data) # Check we haven't lost any rows

#####################################################################################################
# 5. Decision tree model building using training data
#####################################################################################################
# Split data back into training and test sets for model building
training <- full_data[!is.na(full_data$Survived),]
testing <- full_data[is.na(full_data$Survived),]
  
# Convert chracter variables to factors for the model 
training$Survived <- as.factor(training$Survived)
training$FamSizeGroup <- as.factor(training$FamSizeGroup)
training$AgeGroup <- as.factor(training$AgeGroup)
training$Embarked <- as.factor(training$Embarked)
training$FareGroup <- as.factor(training$FareGroup)

testing$Survived <- as.factor(testing$Survived)
testing$FamSizeGroup <- as.factor(testing$FamSizeGroup)
testing$AgeGroup <- as.factor(testing$AgeGroup)
testing$Embarked <- as.factor(testing$Embarked)
testing$FareGroup <- as.factor(testing$FareGroup)
str(training)
str(testing)

# Build decision tree model with rpart 
tree_model <- rpart(Survived ~ AgeGroup + FareGroup + Pclass + Sex + Title + FamSizeGroup + Parch + SibSp, method = "class", data = training)

# Visualise models 
rpart.plot(tree_model, type = 3, box.palette = c("red", "green"), fallen.leaves = TRUE)

# Model is complex at the moment, so let's determine complexity parameters to prune it with
plotcp(tree_model)

# Now use that 0.012 value to prune the model back and reduce complexity
pruned_model <- prune(tree_model, cp = 0.02)

# Visualise pruned model
rpart.plot(pruned_model, type = 3, box.palette = c("red", "green"), fallen.leaves = TRUE)

####################################################################################################
# 6. Predicting survival using test data with tree model
####################################################################################################
# Predict survival using original model
tree_pred <- predict(pruned_model, testing, type = "class")

# Let's view the predictions and create the Kaggle submission
tree_predictions <- data.frame(PassengerId = testing$PassengerId, Survived = tree_pred)
write.csv(predictions, file = "KaggleSubmission2.csv", row.names = FALSE) # 0.77 Kaggle score

#################################################################################################
# 7. Applying Random Forests algorithmn to the problem as ensemble method
################################################################################################
# Let's see if we can improve the prediction results by using randomForests over basic tree model
# Build randomForest classification model to model survival
set.seed(990)
survival_model <- randomForest(as.factor(Survived) ~ AgeGroup + Pclass + Title + Sex + SibSp + FamSizeGroup + Parch + Embarked + FareGroup, data = training, importance = TRUE, ntree = 2000)

# Which variables were most important in the model?
# Mean Decrease Accuracy - shows how much worse model performs when that variable is removed, eg. does it have an impact?
# Mean Decrease Gini - measures purity of each node by seeing what happens when each variable is removed - high score = important variable
varImpPlot(survival_model) # Looks like SibSp, Parch & Embarked are less important in prediction here

################################################################################################
# 8. Predicting survival using test data with randomForest model
################################################################################################
# Prediction
forest_pred <- predict(survival_model, testing)

# Create dataframe for the Kaggle submission
forest_predictions <- data.frame(PassengerId = testing$PassengerId, Survived = forest_pred)
write.csv(forest_predictions, file = "KagglesubmissionForest1.csv", row.names = FALSE) # 0.78 Kaggle score

####################################################################################################
# 9. Conditional inference forest model building
###################################################################################################
# Build conditional inference forest model
set.seed(000)
ci_forest <- cforest(as.factor(Survived) ~ AgeGroup + Pclass + Title + Sex + SibSp + FamSizeGroup + Parch + Embarked + FareGroup, data = training, controls = cforest_unbiased(ntree = 2000, mtry = 3))

# Predictions
ciforest_pred <- predict(ci_forest, testing, OOB = TRUE, type = "response")

# Let's create the output for the Kaggle submission
ci_preds <- data.frame(PassengerId = testing$PassengerId, Survived = ciforest_pred)
write.csv(ci_preds, file = "kagglesubmissionCIForest1.csv", row.names = FALSE) # 0.80 Kaggle score

#################################################################################################
# 10. Improving on the CI Forest model
################################################################################################
# Are there any weakly predictive features we should just remove?
varImpPlot(survival_model) # Embarked, SibSp, & Parch don't fare well

# What happens if we sub out the above features and add in 'Family' (ID coded family groupings)?
# Make sure the Family variable is a factor first
training$Family <- as.factor(training$Family)
testing$Family <- as.factor(testing$Family)

# Build new conditional inference forest model
set.seed(321)
ci_forest2 <- cforest(as.factor(Survived) ~ AgeGroup + Pclass + Title + Sex + Family + FamSizeGroup + FareGroup, data = training, controls = cforest_unbiased(ntree = 2000, mtry = 3))

# Make new set of predictions, and create output file for Kaggle submission
updated_pred <- predict(ci_forest2, testing, OOB = TRUE, type = "response")
