# install library
library(tidyverse)
install.packages("janitor")
library(janitor)
library(scales)
install.packages("rio")
library(rio)
install.packages("gbm")
library(gbm)
library(caret)
install.packages("RANN")
library(RANN)
install.packages("tictoc")
library(tictoc)

# Start run
tic()

# import dataset

heart = read.csv("heart.csv")

##add id var, look for blanks, re-code ExerciseAngina

heart = heart %>% 
  mutate(id = row_number(),
         across(everything(), ~ifelse(.x == "", NA, .x)),
         ExerciseAngina = case_when(ExerciseAngina == "Y" ~ 1,
                                     ExerciseAngina == "N" ~ 0))
slice_head(heart, n = 5)

# Check NA Values * 0 NA Values
nas = anti_join(heart, na.omit(heart), by = "id")
na_rows = nrow(nas)
na_rows

#summarization function for strings

counter = function(df, varname) {
  df %>% 
    count({{varname}}) %>% 
    mutate(pct = round(n / sum(n), 2)) %>% 
    arrange(-n)
}

#apply

counter(heart, Sex)
counter(heart, ChestPainType)
counter(heart, RestingECG)
counter(heart, ExerciseAngina)
counter(heart, ST_Slope)

#target var split
target_split = counter(heart, HeartDisease)


#baseline for prediction (to compare to 'just guessing')
baseline = (target_split %>% filter(HeartDisease == 1))$pct

#distributions of continuous variables
hist_fn = function(varname) {
  ggplot(data = heart,
         aes(x = {{varname}})) +
    geom_histogram() +
    labs(title = "Variable distribution",
         y = "Frequency") +
    scale_y_continuous(labels = comma)
}

hist_fn(RestingBP)
hist_fn(Cholesterol) 
hist_fn(log(Cholesterol))
hist_fn(MaxHR)
hist_fn(Oldpeak)

#create ID name
heart$heart_disease_char = ifelse(heart$HeartDisease == 1, "Heart Disease", "No Heart Disease")

#summary table for continuous variables
heart %>%
  select(heart_disease_char, Age, RestingBP, Cholesterol, FastingBS, MaxHR, ExerciseAngina, Oldpeak) %>% 
  group_by(heart_disease_char) %>% 
  summarise_all(.funs = "mean")
#remove unnecessary variables
heart = heart %>% select(!c(id, heart_disease_char))

#recode zeros to NA for specific columns
heart = heart %>% 
  mutate(across(c(RestingBP, Cholesterol), ~ifelse(.x == 0, NA, .x)))

#use caret preProcess to impute the median value for the one missing value in resting_bp
heart = predict(preProcess(heart %>% select(RestingBP),
                           method = "medianImpute"), 
                heart)

#for cholesterol, use bag imputation since there's quite a lot of missing data
heart = predict(preProcess(heart,
                           method = "bagImpute"),
                heart)

#re-plot cholesterol
hist_fn(Cholesterol)

#check out new cholestrol averages by target variable
heart %>% 
  group_by(HeartDisease) %>% 
  summarise(avg_cholesterol = mean(Cholesterol, na.rm = T))

#one hot encode strings
dummy = c("Sex", "ChestPainType", "RestingECG", "ST_Slope")

#create df for dummies and others
dummy_df = heart %>% select(all_of(dummy))
heart_df = heart %>% select(!all_of(dummy))

#dummify data, join back to main dataset
heart = data.frame(predict(dummyVars(" ~ .", 
                                     data = dummy_df),
                           newdata = dummy_df)) %>% 
  clean_names() %>% 
  bind_cols(heart_df)

#rescale and center variables
heart = predict(preProcess(heart %>% 
                             select(Age, RestingBP, Cholesterol, MaxHR, Oldpeak),
                           method = c("center", "scale")), 
                heart)

#remove unnecessary stuff
rm(dummy_df, heart_df, nas, target_split)

#correlations
correlations = data.frame(cor(heart)) %>% 
  mutate(across(everything(), ~round(.x, 2))) 

#create name column, remove names from index
correlations$x = row.names(correlations)
row.names(correlations) = NULL

#re-arrange data frame
correlations = correlations %>% 
  relocate(x) %>% 
  pivot_longer(!x,
               names_to = "y",
               values_to = "r")

#look at highly correlated variables
correlations %>% 
  mutate(same = ifelse(x == y, 1, 0)) %>% 
  filter(same == 0,
         abs(r) > 0.7)

#remove highly correlative / unnecessary / duplicative columns, recode outcome variable
heart = heart %>% 
  select(!c(sex_m, st_slope_up)) %>% 
  mutate(HeartDisease = ifelse(HeartDisease == 1, "Y", "N"),
         HeartDisease = as.factor(HeartDisease))

#split data into training and test sets
split = createDataPartition(heart$HeartDisease, p = .7, list = F)
train = data.frame(heart[ split, ])
test = data.frame(heart[-split, ])

#confirm class balances compared to baseline
counter(train, HeartDisease)
counter(test, HeartDisease)
baseline

#stochastic gradient boosting
control_gbm = trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 10)

grid_gbm = expand.grid(interaction.depth = c(1, 3, 5),
                       n.trees = (1:10) * 10,
                       shrinkage = 0.1,
                       n.minobsinnode = 10)

model_gbm = train(HeartDisease ~ ., 
                  data = train,
                  method = "gbm",
                  tuneGrid = grid_gbm,
                  trControl = control_gbm,
                  verbose = FALSE)

#in sample results
ggplot(model_gbm) +
  labs(title = "Results from in-sample model training",
       subtitle = "Model type: gradient boosting") +
  scale_y_continuous(labels = percent)

#return best model parameters
model_gbm$bestTune

#feature importance
importance = varImp(model_gbm)
ggplot(importance)

#out of sample predictions, results, accuracy
yhat_gbm = predict(model_gbm, test)
cm_gbm = confusionMatrix(test$HeartDisease, yhat_gbm, positive = "Y")
cm_gbm

#End run
toc()