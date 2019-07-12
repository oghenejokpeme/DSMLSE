library(caret)
library(kernlab)
library(xgboost)
library(parallel)
library(doMC)

getTestIndices <- function(sample_number, percent_split = 0.25){
  sample_size <- floor(percent_split * sample_number)
  set.seed(23452)
  test_indices <- sample(seq_len(sample_number), size = sample_size)
  
  return(test_indices)
}

generateCvFolds <- function(n, folds = 5){
  set.seed(13579)
  folds <- split(sample(1:n), rep(1:folds, length = n))

  return(folds)
}

getPerformanceMetrics <- function(y_true, pred){
  rsquared <- 1 - (sum((y_true - pred)^2) / sum((y_true - mean(y_true))^2))
  mse <- sum((y_true - pred)^2) / length(y_true)
  rmse <- sqrt(sum((y_true - pred)^2) / length(y_true))
  perfs <- c(rsquared = rsquared, mse = mse, rmse = rmse)
  
  return(perfs)
}

getSplitData <- function(){
  x <- read.csv(file = "../input/data/processed/x.csv", header = T, 
                row.names = 1)
  ys <- read.csv(file = "../input/data/processed/ys.csv", header = T,
                 row.names = 1)
  x <- data.matrix(x)
  ys <- data.matrix(ys) 
  test_indices <- getTestIndices(dim(x)[1])
  
  x_train <- x[-test_indices, ]
  ys_train <- ys[-test_indices, ]
  x_test <- x[test_indices, ]
  ys_test <- ys[test_indices, ]
  
  data <- list(x_train = x_train, ys_train = ys_train,
               x_test = x_test, ys_test = ys_test)
  
  return(data)
}

readChrSnps <- function(chr){
  snps <- as.matrix(read.table(file = paste0("../input/all_snps/", chr, ".txt"), 
                               header = F)
                   )[,1]
  
  return(snps)
}

logModel <- function(trait, learner, exp, model, fold = NULL, chr = NULL){
  path <- ""
  if (!is.null(fold) && is.null(chr)){
    path <- paste0("../output/models/", exp, "/", trait, "_", learner, "_",
                   fold, ".rds")
  } else if (!is.null(fold) && !is.null(chr)){
    path <- paste0("../output/models/", exp, "/", trait, "_", learner, "_",
                   chr, "_", fold, ".rds")
  } else {
    path <- paste0("../output/models/", exp, "/", trait, "_", learner, ".rds")
  }
  
  saveRDS(model, path)
}

getModel <- function(trait, learner, exp, fold = NULL, chr = NULL){
  path <- ""
  if (!is.null(fold) && is.null(chr)){
    path <- paste0("../output/models/", exp, "/", trait, "_", learner, "_",
                   fold, ".rds")
  } else if (!is.null(fold) && !is.null(chr)){
    path <- paste0("../output/models/", exp, "/", trait, "_", learner, "_",
                   chr, "_", fold, ".rds")
  } else {
    path <- paste0("../output/models/", exp, "/", trait, "_", learner, ".rds")
  }
  
  readRDS(path)
}

logPredictions <- function(trait, set, df, fold = NULL, chr = NULL){
  path <- ""
  if (!is.null(fold) && is.null(chr)){
    path <- paste0("../output/predictions/fa/", trait, "_", set, "_", fold, 
                   ".csv")
  } else if (!is.null(fold) && !is.null(chr)) {
    path <- paste0("../output/predictions/fb/", trait, "_", set, "_", chr, 
                   "_", fold, ".csv")
  }
  
  write.csv(df, path)
}

getPredictions <- function(trait, set, fold = NULL, chr = NULL){
  path <- ""
  if (!is.null(fold) && is.null(chr)){
    path <- paste0("../output/predictions/fa/", trait, "_", set, "_", fold, 
                   ".csv")
  } else if (!is.null(fold) && !is.null(chr)) {
    path <- paste0("../output/predictions/fb/", trait, "_", set, "_", chr, 
                   "_", fold, ".csv")
  }
  df <- read.csv(path, header = T, row.names = 1)

  return(df)
}

logPerf <- function(trait, learner, exp, perfs){
  path <- paste0("../output/results/", exp, "/", learner, ".txt")
  perf <- c(trait, perfs)
  write(perf, ncolumns = length(perf), append = T, path)
}

cvKsvm <- function(x, y, params, fold_idxs){
  sigma <- params[1]
  C <- params[2]
  epsilon <- params[3]
  cvperfs <- NULL
  for (fold_idx in fold_idxs){
    x_train <- x[-fold_idx, ]
    y_train <- y[-fold_idx]
    x_test <- x[fold_idx, ]
    y_test <- y[fold_idx]
    set.seed(4453)
    svr_mod <- ksvm(x = x_train, y = y_train, scaled = FALSE, 
                    kernel = "rbfdot", type = "eps-svr", C = C,
                    epsilon = epsilon, kpar = list(sigma = sigma))
    svr_pred <- predict(svr_mod, x_test)[, 1]
    rmse <- getPerformanceMetrics(y_test, svr_pred)["rmse"]
    cvperfs <- c(cvperfs, rmse)
  }
  
  return(mean(cvperfs))
}

fitSvr <- function(x, y, inparallel = TRUE){
  folds <- 3
  grid <- expand.grid(sigma = 2^seq(-15, 1, by = 2),
                      C = c(0.1, 1, 10, 100),
                      epsilon = c(0.001, 0.01, 0.1)
                     )
  grid <- expand.grid(sigma = 2^seq(-15, 1, by = 2),
                      C = c(0.1, 1, 10, 100),
                      epsilon = c(0.1)
                     )
  
  rmses <- NULL
  fold_idxs <- generateCvFolds(length(y), folds)
  if (!inparallel){
    rmses <- apply(grid, 1, function(params){
                   params <- unlist(params)
                   cvKsvm(as.matrix(x), y, params, fold_idxs)
                  })
  
  } else {
    rmses <- foreach(i = 1:dim(grid)[1], .combine = 'c',
                     .inorder = TRUE) %dopar% {
                     params <- unlist(grid[i, ])
                     cvKsvm(as.matrix(x), y, params, fold_idxs)
             }
  }
  
  best_params_idx <- as.vector(which(rmses == min(rmses)))[1]
  best_params <- unlist(grid[best_params_idx, ])
  sigma <- best_params[1]
  C <- best_params[2]
  epsilon <- best_params[3]   
  set.seed(4453)
  svr_fit <- ksvm(x = x, y = y, scaled = FALSE, 
                  kernel = "rbfdot", type = "eps-svr", C = C,
                  epsilon = epsilon, kpar = list(sigma = sigma))
  svr_model <- list(perfs = cbind(grid, rmses),
                    finalModel = svr_fit)
  
  return(svr_model)
}

fitXgBoost <- function(x, y){
  if (getDoParRegistered() == TRUE){
    registerDoSEQ()
  }
  ctrl <- trainControl(method = "cv", number = 3, allowParallel = FALSE)
  grid <- expand.grid(nrounds = c(500, 1000, 1500, 2000),
                      max_depth = c(3, 5, 7, 10),
                      eta = 0.01,
                      gamma = c(0.05, 0.1),
                      subsample = 0.5,
                      colsample_bytree = 0.5,
                      min_child_weight = c(1, 5, 10)
                     )
  set.seed(47567)
  xgb_fit <- train(x = x, y = y, method = "xgbTree", 
                   verbose = 0, eval_metric = "rmse", trControl = ctrl,
                   tuneGrid = grid)
  
  return(xgb_fit)
}

LrWeights <- function(df){
  lm_fit <- lm(y~. + 0, data = df)
  weights <- as.vector(coef(lm_fit))

  return(weights)
}
