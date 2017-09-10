# These data were obtained from https://www.kaggle.com/c/titanic 

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
library(mice) # Missing value imputation
library(dplyr) # Data manipulation/cleaning
library(randomForest) # randomForest model
library(ggplot2) # Visualisation 

# Load Titanic survival training & test datasets
# Dataset was split into test and training automatically by Kaggle.com 
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

unique(full_data$Title) # We now have 5 unque titles to work with!

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
##################################################################################################

## UNFINISHED ##



#####################################################################################################
# 4. Decision tree model building using training data
#####################################################################################################
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

####################################################################################################
# 5. Predicting survival using test data 
####################################################################################################
# Predict survival using original model
pred <- predict(survival_model, testing, type = "class")

# Predict survival using simplified (pruned) tree model
pred2 <- predict(survival_model2, testing, type = "class")

# Performance assessment - create confusion matrix to compare predicted and actual survival
##table(loans_test$pred, loans_test$outcome)

# Compute the accuracy of testing dataset - gives a % accuracy score
##mean(loans_test$pred == loans_test$outcome)  

#################################################################################################
# 6. Applying Random Forests algorithmn to the problem as ensemble method
################################################################################################
# Build forest
#### object <- randomForest(outcome_var ~ var2 + var3, data = data, ntree = 500, mtry = sqrt(p))
# use predict and performance check as normal
