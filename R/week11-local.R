# Script Setting and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(haven)
library(tidyverse)
library(caret)
library(parallel)
library(doParallel)
library(tictoc)
library(janitor)

# Data Import and Cleaning
gss_data <- read_spss("../data/GSS2016.sav")
gss_tbl <- gss_data %>%
  filter(!is.na(MOSTHRS)) %>%
  rename(`work hours` = MOSTHRS) %>%
  select(-HRS1, -HRS2) %>%
  remove_empty("cols", cutoff = 0.25) %>%  
  sapply(as.numeric) %>%
  as_tibble()

# Visualization
gss_tbl %>%
  ggplot(aes(x = `work hours`)) +
  geom_histogram()

# Analysis
set.seed(123) 

mod_vec = c("lm", "glmnet", "ranger", "xgbTree")
index = createDataPartition(gss_tbl$`work hours`, p = 0.75, list = FALSE)
gss_tbl_train = gss_tbl[index,]
gss_tbl_test = gss_tbl[-index,]

training_folds = createFolds(gss_tbl_train$`work hours`, 10)

reuseControl = trainControl( method = "cv", number = 10, search = "grid", 
                             indexOut = training_folds, verboseIter = TRUE)
# Initialize storage for models and timing
mod_ls_original <- list()
mod_ls_parallel <- list()
original_time <- rep(0, length(mod_vec)) # Storage for original timing
parallel_time <- rep(0, length(mod_vec)) # Storage for parallel timing

# Original training (without parallelization)
for(i in 1:length(mod_vec)){
  method = mod_vec[i]
  if(method == "lm" | method == "glmnet"){
    pre_process = c("center", "scale", "nzv", "medianImpute")
  }else{
    pre_process = "medianImpute"
  }
  # tictoc for timing
  tic()
  mod = train(`work hours` ~ .,
              data = gss_tbl_train,
              method = method,
              metric = "Rsquared",
              na.action = na.pass,
              trControl = reuseControl,
              preProcess = pre_process)
  time_store = toc()
  time_elapsed = time_store$toc - time_store$tic # computing seconds elapsed
  original_time[i] = time_elapsed # storing seconds elapsed 
  mod_ls_original[[i]] = mod
}


# Parallel training (with parallelization)
local_cluster <- makeCluster(detectCores() - 1)   
registerDoParallel(local_cluster)

for(i in 1:length(mod_vec)){
  method = mod_vec[i]
  if(method == "lm" | method == "glmnet"){
    pre_process = c("center", "scale", "nzv", "medianImpute")
  }else{
    pre_process = "medianImpute"
  }
  # Added tic(), toc() to capture time needed to train a model 
  tic()
  mod = train(`work hours` ~ .,
              data = gss_tbl_train,
              method = method,
              metric = "Rsquared",
              na.action = na.pass,
              trControl = reuseControl,
              preProcess = pre_process)
  time_store = toc()
  time_elapsed = time_store$toc - time_store$tic # computing seconds elapsed
  parallel_time[i] = time_elapsed # storing seconds elapsed
  mod_ls_parallel[[i]] = mod
}

stopCluster(local_cluster) # End parallel processing
registerDoSEQ() # Switch back to sequential processing

# Publication
## Create a function to aggregate results
results <- function(train_mod){
  algo <- train_mod$method
  cv_rsq <- str_remove(format(round(max(train_mod$results$Rsquared), 2), nsmall = 2), "^0")
  preds <- predict(train_mod, gss_tbl_test, na.action = na.pass)
  ho_rsq <- str_remove(format(round(cor(preds, gss_tbl_test$`work hours`)^2, 2), nsmall = 2), "^0")
  return(c("algo" = algo, "cv_rsq" = cv_rsq, "ho_sq" = ho_rsq))
}

# Publication
table1_tbl <- as_tibble(t(sapply(mod_ls_original, results))) 
table2_tbl <- tibble(algorithm = mod_vec,
                    original = original_time,
                    parallelized = parallel_time)
# XGBoost benefited most from parallelization, with training time reduced significantly, thanks to its efficient use of parallel computation. GLMNet also showed considerable improvement, highlighting the effectiveness of parallel processing in lambda selection
# The difference between the fastest (GLMNet, 4.023 seconds) and the slowest (Ranger, 77.858 seconds) parallelized model was 73.835 seconds, attributed to the computational complexity of the algorithms and their parallelization capabilities
# Considering both performance and computational efficiency, XGBoost is recommended for production due to its superior predictive performance and significant training time reduction when parallelized


