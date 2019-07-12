source("exp_lib.R")

logChainModel <- function(trait, chain_idx, chain, learner, model){
  path <- paste0("../output/models/erc/", trait, "_", learner, 
                 "_", chain_idx, ".rds")
  chain_model <- list(chain = chain, chain_idx = chain_idx, 
                      model = model)
  
  saveRDS(chain_model, path)
}

getChainModel <- function(trait, chain_idx, learner){
  path <- paste0("../output/models/erc/", trait, "_", learner, 
                 "_", chain_idx, ".rds")
  
  readRDS(path)
}

logChainModelPrediction <- function(chain_idx, learner, df){
  path <- paste0("../output/predictions/erc/", learner, "_", 
                 chain_idx, ".csv")
  write.csv(df, path)
}

getChainModelPrediction <- function(chain_idx, learner){
  path <- paste0("../output/predictions/erc/", learner, "_", 
                 chain_idx, ".csv")
  read.csv(path, header = TRUE, row.names = 1)
}

buildChainModels <- function(df, trait_chains, inparallel){
  x_train <- df$x_train
  ys_train <- df$ys_train
  x_test <- df$x_test
  ys_test <- df$ys_test
   
  for (ci in 1:length(trait_chains)) {
    print(paste0("Chain index: ", ci))
    trait_chain <- trait_chains[[ci]]
    print(trait_chain)
    for (trait in trait_chain){
      if (inparallel){
        registerDoMC(cores = 20)
      }
      nx_train <- x_train
      y_train <- ys_train[, trait]
      trait_idx <- which(trait_chain == trait)
      atraits <- NULL
      print(paste0(trait, ", TI: ", trait_idx))
      if (trait_idx > 1){
        atraits <- trait_chain[1:trait_idx-1]
        trait_df <- as.matrix(ys_train[, atraits])
        colnames(trait_df) <- atraits
        nx_train <- cbind(x_train, trait_df)
      }

      print("  SVR")
      svr_model <- fitSvr(nx_train, y_train, inparallel)
      logChainModel(trait, ci, trait_chain, "svr", svr_model)
      print("  XGB")
      xgb_model <- fitXgBoost(nx_train, y_train)
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
    logPerf(trait, "svr", "erc", svr_perf)
    
    xgb_perf <- getPerformanceMetrics(y_test, fxgb)
    logPerf(trait, "xgb", "erc", xgb_perf)
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
  path <- "../output/logs/erc_chain.txt"
  for (i in 1:length(chains)){
    chain <- chains[[i]]
    write(chain, ncolumns = length(chain), append = TRUE,
          file = path)
  }
}

runExperiments <- function(nchains = 10, inparallel = TRUE){ 
  df <- getSplitData()
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
