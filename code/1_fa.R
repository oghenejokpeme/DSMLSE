source("exp_lib.R")

createMetaFeatures <- function(df, vfolds, inparallel){
  x_learn <- df$x_train
  ys_learn <- df$ys_train
  x_test <- df$x_test
  ys_test <- df$ys_test
   
  fold_idxs <- generateCvFolds(dim(x_learn)[1], vfolds)
  for (trait in colnames(ys_learn)){
    print(trait)
    y_learn <- ys_learn[, trait]
    y_test <- ys_test[, trait]
    
    for (fold in 1:length(fold_idxs)){
      if (inparallel){
        registerDoMC(cores = 20) 
      }
      fold_idx <- fold_idxs[[fold]]
      
      x_train <- x_learn[-fold_idx, ]
      y_train <- y_learn[-fold_idx]
      x_val <- x_learn[fold_idx, ]
      y_val <- y_learn[fold_idx]
     
      print(paste0("  Fold: ", fold))
      print("  SVR") 
      svr_model <- fitSvr(x_train, y_train, inparallel)
      logModel(trait, "svr", "fa", svr_model, fold)
      print("  XGB") 
      xgb_model <- fitXgBoost(x_train, y_train)
      logModel(trait, "xgb", "fa", xgb_model, fold)
      
      svr_val_pred <- predict(svr_model$finalModel, x_val)[, 1]
      xgb_val_pred <- predict(xgb_model$finalModel, x_val)
      val_pred <- cbind(svr = svr_val_pred, xgb = xgb_val_pred, y = y_val)
      logPredictions(trait, "val", val_pred, fold)
            
      svr_test_pred <- predict(svr_model$finalModel, x_test)[, 1]
      xgb_test_pred <- predict(xgb_model$finalModel, x_test)
      test_pred <- cbind(svr = svr_test_pred, xgb = xgb_test_pred)
      rownames(test_pred) <- rownames(x_test)
      logPredictions(trait, "test", test_pred, fold)
    }
  }
}  

getMergedTestSet <- function(trait, vfolds){
  svr_df <- NULL
  xgb_df <- NULL
  for (fold in 1:vfolds){
    df <- getPredictions(trait, "test", fold)
    svr_df <- cbind(svr_df, df[, 1])
    xgb_df <- cbind(xgb_df, df[, 2])
  }
  df <- cbind(svr = rowMeans(svr_df), xgb = rowMeans(xgb_df))
  
  return(df)
}

getWeights <- function(trait, vfolds){
  all_weights <- list()
  for (fold in 1:vfolds){
    df <- getPredictions(trait, "val", fold)
    all_weights[[fold]] <- LrWeights(df)
  }
  
  return(all_weights)
}

makeStdPrediction <- function(trait, vfolds, y_test){
  x_test_avg <- getMergedTestSet(trait, vfolds)
  all_weights <- getWeights(trait, vfolds)
  
  preds <- NULL
  for (fold in 1:vfolds){
    weights <- as.vector(all_weights[[fold]])
    pred <- rowSums(x_test_avg %*% diag(as.vector(weights)))
    preds <- cbind(preds, pred)
  }
  fhat <- rowMeans(preds)
  perf <- getPerformanceMetrics(y_test, fhat)
  
  return(perf)
}

faLogPerf <- function(trait, type, perf){
  path <- paste0("../output/results/fa/", type, ".txt")
  rperf <- c(trait, perf)
  write(rperf, ncolumns = length(rperf), append = TRUE, path)
}

combineMetaFeatures <- function(df, vfolds){
  ys_test <- df$ys_test
  
  for (trait in colnames(ys_test)){
    print(trait)
    y_test <- ys_test[, trait]
    std_perf <- makeStdPrediction(trait, vfolds, y_test)
    faLogPerf(trait, type = "std", std_perf)
  }
}

runExperiments <- function(vfolds = 5, inparallel = TRUE){ 
  df <- getSplitData()
  print("Creating meta features...")
  createMetaFeatures(df, vfolds, inparallel)
  print("Combining meta features...")
  combineMetaFeatures(df, vfolds) 
}

runExperiments()
