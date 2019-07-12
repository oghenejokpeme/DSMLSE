source("exp_lib.R")

createCorrectedInputs <- function(df, inparallel){
  ffolds <- 5
  x_learn <- df$x_train
  ys_learn <- df$ys_train

  fold_idxs <- generateCvFolds(dim(x_learn)[1], ffolds)
  svr_trait_ndf <- NULL
  xgb_trait_ndf <- NULL
  for (trait in colnames(ys_learn)){
    y_learn <- ys_learn[, trait]
    print(trait)
    svr_ndf <- NULL
    xgb_ndf <- NULL
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
      
      svr_pred <- as.matrix(predict(svr_model$finalModel, x_val))
      xgb_pred <- as.matrix(predict(xgb_model$finalModel, x_val))
      
      rownames(svr_pred) <- rownames(x_learn)[fold_idx]
      rownames(xgb_pred) <- rownames(x_learn)[fold_idx]
      
      svr_ndf <- rbind(svr_ndf, svr_pred)
      xgb_ndf <- rbind(xgb_ndf, xgb_pred)
    }
    svr_ndf <- svr_ndf[rownames(x_learn), ]
    xgb_ndf <- xgb_ndf[rownames(x_learn), ]
    
    svr_trait_ndf <- cbind(svr_trait_ndf, svr_ndf)
    xgb_trait_ndf <- cbind(xgb_trait_ndf, xgb_ndf)
  }
  colnames(svr_trait_ndf) <- colnames(ys_learn)
  colnames(xgb_trait_ndf) <- colnames(ys_learn)
  
  write.csv(svr_trait_ndf, file = "../output/predictions/ercc/svr_corrected.csv")
  write.csv(xgb_trait_ndf, file = "../output/predictions/ercc/xgb_corrected.csv")
}

logChainModel <- function(trait, chain_idx, chain, learner, model){
  path <- paste0("../output/models/ercc/", trait, "_", learner, 
                 "_", chain_idx, ".rds")
  chain_model <- list(chain = chain, chain_idx = chain_idx, 
                      model = model)
  
  saveRDS(chain_model, path)
}

getChainModel <- function(trait, chain_idx, learner){
  path <- paste0("../output/models/ercc/", trait, "_", learner, 
                 "_", chain_idx, ".rds")
  
  readRDS(path)
}

logChainModelPrediction <- function(chain_idx, learner, df){
  path <- paste0("../output/predictions/ercc/", learner, "_", 
                 chain_idx, ".csv")
  write.csv(df, path)
}

getChainModelPrediction <- function(chain_idx, learner){
  path <- paste0("../output/predictions/ercc/", learner, "_", 
                 chain_idx, ".csv")
  read.csv(path, header = TRUE, row.names = 1)
}

buildChainModels <- function(df, trait_chains, inparallel){
  x_train <- df$x_train
  ys_train <- df$ys_train
  
  svr_ydf <- read.csv(file = "../output/predictions/ercc/svr_corrected.csv",
                      header = TRUE, row.names = 1)
  xgb_ydf <- read.csv(file = "../output/predictions/ercc/xgb_corrected.csv",
                      header = TRUE, row.names = 1)
                                         
  for (ci in 1:length(trait_chains)) {
    print(paste0("Chain index: ", ci))
    trait_chain <- trait_chains[[ci]]
    print(trait_chain)
    for (trait in trait_chain){
      if (inparallel){
        registerDoMC(cores = 20)
      }
      nx_svr_train <- x_train
      nx_xgb_train <- x_train
      
      y_train <- ys_train[, trait]
      trait_idx <- which(trait_chain == trait)
      atraits <- NULL
           
      if (trait_idx > 1){
        atraits <- trait_chain[1:trait_idx-1]
        svr_trait_df <- as.matrix(svr_ydf[, atraits])
        xgb_trait_df <- as.matrix(xgb_ydf[, atraits])
        
        colnames(svr_trait_df) <- atraits
        colnames(xgb_trait_df) <- atraits
        
        nx_svr_train <- cbind(nx_svr_train, svr_trait_df)
        nx_xgb_train <- cbind(nx_xgb_train, xgb_trait_df)
      }
      
      print("  SVR")
      svr_model <- fitSvr(nx_svr_train, y_train, inparallel)
      logChainModel(trait, ci, trait_chain, "svr", svr_model)
      print("  XGB")
      xgb_model <- fitXgBoost(nx_xgb_train, y_train)
      logChainModel(trait, ci, trait_chain, "xgb", xgb_model)
    }
  }
} 

makeChainPredictions <- function(df, trait_chains){
  x_train <- df$x_train
  ys_train <- df$ys_train
  x_test <- df$x_test
  ys_test <- df$ys_test
    
  for (ci in 1:length(trait_chains)) {
    trait_chain <- trait_chains[[ci]]
    for (trait in trait_chain){
      trait_idx <- which(trait_chain == trait)
      svr_model <- getChainModel(trait, ci, "svr")
      xgb_model <- getChainModel(trait, ci, "xgb")
      
      if (trait_idx == 1){
        svr_pred <- predict(svr_model$model$finalModel, x_test)   
        xgb_pred <- as.matrix(predict(xgb_model$model$finalModel, x_test))
        
        colnames(svr_pred) <- c(trait)
        colnames(xgb_pred) <- c(trait)
        
        rownames(svr_pred) <- rownames(x_test)
        rownames(xgb_pred) <- rownames(x_test)
        
        logChainModelPrediction(ci, "svr", svr_pred)
        logChainModelPrediction(ci, "xgb", xgb_pred)
      
      } else {
        svr_cp <- getChainModelPrediction(ci, "svr")
        xgb_cp <- getChainModelPrediction(ci, "xgb")
        
        x_svr_test <- data.matrix(cbind(x_test, svr_cp))
        x_xgb_test <- data.matrix(cbind(x_test, xgb_cp))
        
        svr_pred <- predict(svr_model$model$finalModel, x_svr_test)   
        xgb_pred <- as.matrix(predict(xgb_model$model$finalModel, x_xgb_test))
        
        colnames(svr_pred) <- c(trait)
        colnames(xgb_pred) <- c(trait)
        
        nsvr_cp <- cbind(svr_cp, svr_pred)
        nxgb_cp <- cbind(xgb_cp, xgb_pred)
        
        logChainModelPrediction(ci, "svr", nsvr_cp)
        logChainModelPrediction(ci, "xgb", nxgb_cp)
      }   
    }
  }
}

mergeChainPredictions <- function(df, trait_chains){
  all_traits <- colnames(df$ys_train)
  
  svr_all <- NULL
  xgb_all <- NULL
  for (ci in 1:length(trait_chains)) {
    svr_cp <- getChainModelPrediction(ci, "svr")
    xgb_cp <- getChainModelPrediction(ci, "xgb")
    for (trait in all_traits){
      tsvr <- svr_cp[, trait]
      txgb <- xgb_cp[, trait]
      
      svr_all[[trait]] <- cbind(svr_all[[trait]], tsvr)
      xgb_all[[trait]] <- cbind(xgb_all[[trait]], txgb)
    }
  }
  
  for (trait in all_traits) {
    y_test <- df$ys_test[, trait]
    fsvr <- rowMeans(svr_all[[trait]])
    fxgb <- rowMeans(xgb_all[[trait]])
    
    svr_perf <- getPerformanceMetrics(y_test, fsvr)
    logPerf(trait, "svr", "ercc", svr_perf)
    
    xgb_perf <- getPerformanceMetrics(y_test, fxgb)
    logPerf(trait, "xgb", "ercc", xgb_perf)
  }
}

createTraitChains <- function(traits, nchains){
  set.seed(64243)
  chains <- NULL
  for (i in 1:nchains){
    chains[[i]] <- sample(traits, length(traits))
  }
  
  return(chains)
}

logTraitChains <- function(chains){
  path <- "../output/logs/ercc_chain.txt"
  for (i in 1:length(chains)){
    chain <- chains[[i]]
    write(chain, ncolumns = length(chain), append = TRUE,
          file = path)
  }
}

runExperiments <- function(nchains = 10, inparallel = TRUE){ 
  df <- getSplitData()
  print("Creating corrected inputs")
  createCorrectedInputs(df, inparallel)
  
  traits <- colnames(df$ys_train)
  trait_chains <- createTraitChains(traits, nchains)
  logTraitChains(trait_chains)
  print("Building base models")
  buildChainModels(df, trait_chains, inparallel)
  print("Making predictions")
  makeChainPredictions(df, trait_chains)
  print("Merging predictions")
  mergeChainPredictions(df, trait_chains)
}

runExperiments()
