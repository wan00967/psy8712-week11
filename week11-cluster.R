# Script Setting and Resources
library(haven)
library(tidyverse)
library(caret)
library(parallel)
library(doParallel)
library(tictoc)
library(janitor)
library(glmnet)

# Data Import and Cleaning
gss_data <- read_spss("data/GSS2016.sav")
gss_tbl <- gss_data %>%
  filter(!is.na(MOSTHRS)) %>%
  rename(`work hours` = MOSTHRS) %>%
  select(-HRS1, -HRS2) %>%
  remove_empty("cols", cutoff = 0.25) %>%  
  sapply(as.numeric) %>%
  as_tibble()

# Analysis
set.seed(123) 

mod_vec = c("lm", "glmnet", "ranger", "xgbTree")
index = createDataPartition(gss_tbl$`work hours`, p = 0.75, list = FALSE)
gss_tbl_train = gss_tbl[index,]
gss_tbl_test = gss_tbl[-index,]

training_folds = createFolds(gss_tbl_train$`work hours`, 10)

reuseControl = trainControl( method = "cv", number = 10, search = "grid", 
                             indexOut = training_folds, verboseIter = TRUE)

mod_ls_original <- list()
mod_ls_parallel <- list()
original_time <- rep(0, length(mod_vec))
parallel_time <- rep(0, length(mod_vec))

for(i in 1:length(mod_vec)){
  method = mod_vec[i]
  if(method == "lm" | method == "glmnet"){
    pre_process = c("center", "scale", "nzv", "medianImpute")
  }else{
    pre_process = "medianImpute"
  }

  tic()
  mod = train(`work hours` ~ .,
              data = gss_tbl_train,
              method = method,
              metric = "Rsquared",
              na.action = na.pass,
              trControl = reuseControl,
              preProcess = pre_process)
  time_store = toc()
  time_elapsed = time_store$toc - time_store$tic 
  original_time[i] = time_elapsed
  mod_ls_original[[i]] = mod
}

local_cluster <- makeCluster(14)   
registerDoParallel(local_cluster)

for(i in 1:length(mod_vec)){
  method = mod_vec[i]
  if(method == "lm" | method == "glmnet"){
    pre_process = c("center", "scale", "nzv", "medianImpute")
  }else{
    pre_process = "medianImpute"
  }
  tic()
  mod = train(`work hours` ~ .,
              data = gss_tbl_train,
              method = method,
              metric = "Rsquared",
              na.action = na.pass,
              trControl = reuseControl,
              preProcess = pre_process)
  time_store = toc()
  time_elapsed = time_store$toc - time_store$tic 
  parallel_time[i] = time_elapsed 
  mod_ls_parallel[[i]] = mod
}

stopCluster(local_cluster)
registerDoSEQ() 

# Publication
results <- function(train_mod){
  algo <- train_mod$method
  cv_rsq <- str_remove(format(round(max(train_mod$results$Rsquared), 2), nsmall = 2), "^0")
  preds <- predict(train_mod, gss_tbl_test, na.action = na.pass)
  ho_rsq <- str_remove(format(round(cor(preds, gss_tbl_test$`work hours`)^2, 2), nsmall = 2), "^0")
  return(c("algo" = algo, "cv_rsq" = cv_rsq, "ho_sq" = ho_rsq))
}

table1_tbl <- as_tibble(t(sapply(mod_ls_original, results))) 
table2_tbl <- tibble(algorithm = mod_vec,
                     supercomputer = original_time, 
                     "supercomputer-14" = parallel_time)

write_csv(table1_tbl, "../out/table3.csv")
write_csv(table2_tbl, "../out/table4.csv")
