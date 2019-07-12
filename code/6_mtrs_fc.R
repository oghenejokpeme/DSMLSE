source("exp_lib.R")

getLearnerMetaFeatures <- function(learner, traits){
  train_df <- NULL
  test_df <- NULL
  for (trait in traits){
    train <- read.csv(paste0("../output/predictions/mtrs/", trait, 
                             "_meta_train.csv"), 
                      header = TRUE, row.names = 1)[, learner]
    test <- read.csv(paste0("../output/predictions/mtrs/", trait, 
                            "_meta_test.csv"), 
                     header = TRUE, row.names = 1)[, learner]
    
    train_df <- cbind(train_df, train)
    test_df <- cbind(test_df, test)
  }
  colnames(train_df) <- traits
  colnames(test_df) <- traits
  
  return(list(train = train_df, test = test_df))
}

stackMetaModels <- function(df, inparallel){
  x_test <- df$x_test
  ys_test <- df$ys_test
 
  traits <- colnames(ys_test)
  svr_df <- getLearnerMetaFeatures("svr", traits)
  xgb_df <- getLearnerMetaFeatures("xgb", traits)
    
  for (trait in traits){
    print(paste0("  ", trait))
    y_test <- ys_test[, trait]

    x_svr_test <- cbind(x_test, svr_df$test)
    x_xgb_test <- cbind(x_test, xgb_df$test)
       
    svr_model <- getModel(trait, "svr_meta", "mtrs")
    xgb_model <- getModel(trait, "xgb_meta", "mtrs")

    svr_pred <- predict(svr_model$finalModel, x_svr_test)[ ,1]   
    xgb_pred <- predict(xgb_model$finalModel, x_xgb_test)
    
    fstack <- rowMeans(cbind(svr_pred, xgb_pred))
    fstack_perf <- getPerformanceMetrics(y_test, fstack) 

    logPerf(trait, "mtrs", "fc", fstack_perf)   
  }
}

runExperiments <- function(inparallel = FALSE){ 
  df <- getSplitData()
  print("Building meta models")
  stackMetaModels(df, inparallel)  
}
