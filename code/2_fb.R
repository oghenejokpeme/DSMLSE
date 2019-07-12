source("exp_lib.R")

createMetaFeatures <- function(df, chrs, vfolds, inparallel){
  x_learn <- df$x_train
  ys_learn <- df$ys_train
  x_test <- df$x_test
  ys_test <- df$ys_test
 
  fold_idxs <- generateCvFolds(dim(x_learn)[1], vfolds)
  for (trait in colnames(ys_learn)){
    print(trait)
    y_learn <- ys_learn[, trait]
    y_test <- ys_test[, trait]
    
    for (chr in 1:chrs){
      print(paste0("Chr: ", chr))
      all_chr_snps <- readChrSnps(chr)
      chr_snps <- intersect(colnames(x_learn), all_chr_snps)
      
      x_chr_learn <- x_learn[, chr_snps]
      x_chr_test <- x_test[, chr_snps]
      print(paste0("  Dim: ", dim(x_chr_learn))) 
      for (fold in 1:length(fold_idxs)){
        if (inparallel){
          registerDoMC(cores = 20) 
        } 
        
        print(paste0("  Fold: ", fold))
        fold_idx <- fold_idxs[[fold]]
        
        x_train <- x_chr_learn[-fold_idx, ]
        y_train <- y_learn[-fold_idx]
        x_val <- x_chr_learn[fold_idx, ]
        y_val <- y_learn[fold_idx]
        
        print("    SVR") 
        svr_model <- fitSvr(x_train, y_train, inparallel)
        logModel(trait, "svr", "fb", svr_model, fold, chr)
        
        print("    XGB") 
        xgb_model <- fitXgBoost(x_train, y_train)
        logModel(trait, "xgb", "fb", xgb_model, fold, chr)
        
        svr_val_pred <- predict(svr_model$finalModel, x_val)[, 1]
        xgb_val_pred <- predict(xgb_model$finalModel, x_val)
        val_pred <- cbind(svr = svr_val_pred, xgb = xgb_val_pred, y = y_val)
        logPredictions(trait, "val", val_pred, fold, chr)
            
        svr_test_pred <- predict(svr_model$finalModel, x_chr_test)[, 1]
        xgb_test_pred <- predict(xgb_model$finalModel, x_chr_test)
        test_pred <- cbind(svr = svr_test_pred, xgb = xgb_test_pred)
        rownames(test_pred) <- rownames(x_test)
        logPredictions(trait, "test", test_pred, fold, chr)
      }
    }
  }
}  

getValFoldPredictions <- function(trait, chrs, fold){
  df <- NULL
  y_true <- NULL
  for (chr in 1:chrs){
    path <- paste0("../output/predictions/fb/", trait, "_val_", chr, 
                   "_", fold, ".csv")
    tdf <- read.csv(path, header = T, row.names = 1)
    x_pred <- tdf[, c(1,2)]
    colnames(x_pred) <- paste(colnames(x_pred), chr, sep = "_")
    y_true <- tdf[, 3] 
    df <- cbind(df, data.matrix(x_pred))   
  }
  df <- cbind(df, y = y_true)
  
  return(df)
}

getMergedValSets <- function(trait, chrs, vfolds){
  val_dfs <- NULL
  for (fold in 1:vfolds){
    fold_df <- getValFoldPredictions(trait, chrs, fold)
    val_dfs[[fold]] <- fold_df
  } 
  
  return(val_dfs)
}

getTestFoldPredictions <- function(trait, chrs, fold){
  df <- NULL
  for (chr in 1:chrs){
    path <- paste0("../output/predictions/fb/", trait, "_test_", chr, 
                   "_", fold, ".csv")
    tdf <- read.csv(path, header = T, row.names = 1)
    colnames(tdf) <- paste(colnames(tdf), chr, sep = "_")
    df <- cbind(df, data.matrix(tdf))   
  }
  
  return(df)
}

getMergedTestSet <- function(trait, chrs, vfolds){
  df <- NULL
  for (chr in 1:chrs){
    svr_chr_df <- NULL
    xgb_chr_df <- NULL
    for (fold in 1:vfolds){
      tdf <- getPredictions(trait, "test", fold, chr)
      svr_chr_df <- cbind(svr_chr_df, tdf[, 1])
      xgb_chr_df <- cbind(xgb_chr_df, tdf[, 2])
    }
    chr_df <- cbind(rowMeans(svr_chr_df), rowMeans(xgb_chr_df))
    colnames(chr_df) <- paste(c("svr", "xgb"), chr, sep = "_")
    df <- cbind(df, chr_df)
  }
  
  return(df)
}

getWeights <- function(val_dfs, vfolds){
  all_weights <- list()
  for (fold in 1:vfolds){
    df <- as.data.frame(val_dfs[[fold]])
    all_weights[[fold]] <- LrWeights(df)
  }
  
  return(all_weights)
}

makeStdPrediction <- function(trait, chrs, vfolds, y_test){
  val_dfs <- getMergedValSets(trait, chrs, vfolds)
  all_weights <- getWeights(val_dfs, vfolds)
  x_test_avg <- getMergedTestSet(trait, chrs, vfolds)
  
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

fbLogPerf <- function(trait, type, perf){
  path <- paste0("../output/results/fb/", type, ".txt")
  rperf <- c(trait, perf)
  write(rperf, ncolumns = length(rperf), append = TRUE, path)
}

combineMetaFeatures <- function(df, chrs, vfolds){
  ys_test <- df$ys_test
  
  for (trait in colnames(ys_test)){
    print(trait)
    y_test <- ys_test[, trait]
    std_perf <- makeStdPrediction(trait, chrs, vfolds, y_test)
    fbLogPerf(trait, type = "std", std_perf)
  }
}

runExperiments <- function(chrs = 12, vfolds = 5, inparallel = TRUE){ 
  df <- getSplitData()
  print("Creating meta features...")
  createMetaFeatures(df, chrs, vfolds, inparallel)
  print("Combining meta features...")
  combineMetaFeatures(df, chrs, vfolds) 
}

runExperiments()
