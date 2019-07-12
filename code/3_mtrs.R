source("exp_lib.R")

createTrainingMetaFeatures <- function(df, inparallel){
  ffolds <- 5
  x_learn <- df$x_train
  ys_learn <- df$ys_train
  x_test <- df$x_test
  ys_test <- df$ys_test
   
  fold_idxs <- generateCvFolds(dim(x_learn)[1], ffolds)
  for (trait in colnames(ys_learn)){
    y_learn <- ys_learn[, trait]
    print(trait)
    trait_ndf <- NULL
    for (fold in 1:ffolds){
      if (inparallel){
        registerDoMC(cores = 20)
      }
      print(paste0("  Fold: ", fold))
      fold_idx <- fold_idxs[[fold]]
      
      x_train <- x_learn[-fold_idx, ]
      y_train <- y_learn[-fold_idx]
      x_val <- x_learn[fold_idx, ]
      
      print("  SVR")
      svr_model <- fitSvr(x_train, y_train, inparallel)
      print("  XGB")
      xgb_model <- fitXgBoost(x_train, y_train)
      
      svr_pred <- predict(svr_model$finalModel, x_val)[, 1]
      xgb_pred <- predict(xgb_model$finalModel, x_val)
      
      fold_df <- cbind(svr = svr_pred, xgb = xgb_pred)
      rownames(fold_df) <- rownames(x_learn)[fold_idx]
      trait_ndf <- rbind(trait_ndf, fold_df)
    }
    trait_ndf <- trait_ndf[rownames(x_learn), ]
    
    write.csv(trait_ndf, file = paste0("../output/predictions/mtrs/",
                                       trait, "_meta_train.csv")
             )
  }
}

buildTestMetaFeatures <- function(df){
  x_test <- df$x_test
  
  for (trait in colnames(df$ys_train)){
    print(paste0("  ", trait))
    svr_model <- getModel(trait, "svr", "base")
    xgb_model <- getModel(trait, "xgb", "base")
    
    svr_pred <- predict(svr_model$finalModel, x_test)[, 1]
    xgb_pred <- predict(xgb_model$finalModel, x_test)
    trait_meta <- cbind(svr = svr_pred, xgb = xgb_pred)
    rownames(trait_meta) <- rownames(x_test)
    
    write.csv(trait_meta, file = paste0("../output/predictions/mtrs/",
                                       trait, "_meta_test.csv")
             )
  }
}

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

buildMetaModels <- function(df, inparallel){
  x_train <- df$x_train
  ys_train <- df$ys_train
  x_test <- df$x_test
  ys_test <- df$ys_test
 
  traits <- colnames(ys_train)
  svr_df <- getLearnerMetaFeatures("svr", traits)
  xgb_df <- getLearnerMetaFeatures("xgb", traits)
  
 
  for (trait in traits){
    print(paste0("  ", trait))
    y_train <- ys_train[, trait]
    y_test <- ys_test[, trait]
    if (inparallel){
      registerDoMC(cores = 20)
    }
 
    x_svr_train <- cbind(x_train, svr_df$train)
    x_xgb_train <- cbind(x_train, xgb_df$train)
    x_svr_test <- cbind(x_test, svr_df$test)
    x_xgb_test <- cbind(x_test, xgb_df$test)
    
    svr_model <- fitSvr(x_svr_train, y_train, inparallel)
    logModel(trait, "svr_meta", "mtrs", svr_model)
    xgb_model <- fitXgBoost(x_xgb_train, y_train)
    logModel(trait, "xgb_meta", "mtrs", xgb_model)
    
    svr_pred <- predict(svr_model$finalModel, x_svr_test)[ ,1]   
    xgb_pred <- predict(xgb_model$finalModel, x_xgb_test)
    
    svr_perf <- getPerformanceMetrics(y_test, svr_pred)
    logPerf(trait, "svr", "mtrs", svr_perf)
    
    xgb_perf <- getPerformanceMetrics(y_test, xgb_pred)
    logPerf(trait, "xgb", "mtrs", xgb_perf)   
  }
}

runExperiments <- function(inparallel = TRUE){ 
  df <- getSplitData()
  print("Building training meta-features")
  createTrainingMetaFeatures(df, inparallel)
  print("Building test meta-features")
  buildTestMetaFeatures(df)
  print("Building meta models")
  buildMetaModels(df, inparallel)  
}

runExperiments()
