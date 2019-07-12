source("exp_lib.R")

buildBaseModels <- function(df, inparallel){
  x_train <- df$x_train
  ys_train <- df$ys_train
  x_test <- df$x_test
  ys_test <- df$ys_test
   
  for (trait in colnames(ys_train)){
    if (inparallel){
      registerDoMC(cores = 20)
    }
    print(trait)
    y_train <- ys_train[, trait]
    y_test <- ys_test[, trait]
    
    print("  SVR") 
    svr_model <- fitSvr(x_train, y_train, inparallel)
    logModel(trait, "svr", "base", svr_model)
    print("  XGB") 
    xgb_model <- fitXgBoost(x_train, y_train)
    logModel(trait, "xgb", "base", xgb_model)
    
    svr_pred <- predict(svr_model$finalModel, x_test)[ ,1]   
    xgb_pred <- predict(xgb_model$finalModel, x_test)
    
    svr_perf <- getPerformanceMetrics(y_test, svr_pred)
    logPerf(trait, "svr", "base", svr_perf)
    
    xgb_perf <- getPerformanceMetrics(y_test, xgb_pred)
    logPerf(trait, "xgb", "base", xgb_perf)
  }
}

runExperiments <- function(inparallel = TRUE){ 
  df <- getSplitData()
  buildBaseModels(df, inparallel)  
}

runExperiments()
