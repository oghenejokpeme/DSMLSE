source("exp_lib.R")

getChainModelPrediction <- function(chain_idx, learner){
  path <- paste0("../output/predictions/ercc/", learner, "_", 
                 chain_idx, ".csv")
  read.csv(path, header = TRUE, row.names = 1)
}

stackChainPredictions <- function(df, trait_chains){
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
    
    fstack <- rowMeans(cbind(fsvr, fxgb))
    fstack_perf <- getPerformanceMetrics(y_test, fstack) 

    logPerf(trait, "ercc", "fc", fstack_perf)
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

runExperiments <- function(nchains = 10, inparallel = FALSE){ 
  df <- getSplitData()
  
  traits <- colnames(df$ys_train)
  trait_chains <- createTraitChains(traits, nchains)
  print("Merging predictions")
  stackChainPredictions(df, trait_chains)
}

runExperiments()
