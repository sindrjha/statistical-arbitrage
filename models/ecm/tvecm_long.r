if (require("tsDyn") == FALSE) {
  install.packages("tsDyn")
  library(tsDyn)
}
if (require("Metrics") == FALSE) {
  install.packages("Metrics")
  library(Metrics)
}
if (require("dplyr") == FALSE) {
  install.packages("dplyr")
  library(dplyr)
}

# Based on the predict.TVAR function from the tsDyn package
predict.TVECM <- function(object, newdata, n.ahead=5, 
                        newdataTrendStart, ...){
  
  ## extract parameters, coefs
  lag <- object$lag
  k <- object$k
  include <- object$include
  B <- object$coeffmat  
  Thresh <- getTh(object)
  nthresh <- object$model.specific$nthresh
  beta <- object$model.specific$beta

  
  ## setup starting values (data in y), innovations (0)
  original.data <- object$model[,1:k, drop=FALSE]
  starting <-   tsDyn:::myTail(original.data,lag)
  innov <- matrix(0, nrow=n.ahead, ncol=k)  

  
  
  if(!missing(newdata)) {
    if(!inherits(newdata, c("data.frame", "matrix","zoo", "ts"))) stop("Arg 'newdata' should be of class data.frame, matrix, zoo or ts")
    if(nrow(newdata)!=lag+1) stop(paste0("Please provide newdata with nrow=lag=", lag+1))
    starting <-  as.matrix(newdata)
  }
  
  ## trend DOES NOT WORK YET
  #if(missing(newdataTrendStart)){
  #  if(include%in%c("trend", "both")){
  #    trendStart <- object$t+1
  #  }  else {
  #    trendStart <- 0
  #  }
  #} else {
  #  trendStart <- newdataTrendStart
  #}
  
  
  res <- tsDyn:::TVECM.gen(B=B, beta=beta, n=n.ahead, lag=lag, 
                      include = include, 
                      nthresh= nthresh,
                      Thresh = Thresh, 
                      starting=starting, innov=innov)
                      
  if (n.ahead == 1) {
    res <- matrix(res, nrow = 1)  # Convert vector to matrix with 1 row
  }

  # Format results by assigning column names
  colnames(res) <- colnames(original.data)
  ## format results
  colnames(res) <- colnames(original.data)
  end_rows <- nrow(original.data) + n.ahead
  if(hasArg("returnStarting") && isTRUE(list(...)["returnStarting"])) {
    start_rows <- nrow(original.data)+1 - lag
  } else {
    start_rows <- nrow(original.data)+1
  }
  rownames(res) <- start_rows : end_rows
  
  return(res)
}

tvecm_test_predictions <- function(hubs, window_size = 5, test_size = 250, nthresh = 1, lags) {
  # Initialize empty data frames for storing predictions and actual values
  predictions <- data.frame(matrix(ncol = ncol(hubs), nrow = 0))
  last_available <- data.frame(matrix(ncol = ncol(hubs), nrow = 0))
  actuals <- data.frame(matrix(ncol = ncol(hubs), nrow = 0))
  
  colnames(predictions) <- colnames(hubs)
  colnames(last_available) <- colnames(hubs)
  colnames(actuals) <- colnames(hubs)

  
  for (i in 1:test_size) {

    train_size <- nrow(hubs) - test_size - window_size + i
    hub_train <- hubs[1:train_size, ]

    tvec <- TVECM(hub_train, include = "const", nthresh=nthresh,lag=lags, ngridBeta=100, ngridTh=150, plot=TRUE,trim=0.05, common="All")

    start <- train_size - lags
    end <- train_size

    pred_data <- hubs[start:end, ]

    hub_forecast <- predict.TVECM(tvec, newdata=pred_data, n.ahead=window_size)
    
    hub_prediction <- hub_forecast[window_size, , drop = FALSE]

    hub_last_available <- hubs[end, ]
    hub_actual <- hubs[end + window_size, ]
    
    predictions <- rbind(predictions, hub_prediction)
    last_available <- rbind(last_available, hub_last_available)
    actuals <- rbind(actuals, hub_actual)
  }
  
  # Return both data frames as a list
  return(list(predictions = predictions, actuals = actuals, last_available = last_available))
}

tvecm_validation_predictions <- function(hubs, window_size = 5, validation_size = 250, test_size = 250, nthresh=1, lags) {
  # Initialize empty data frames for storing predictions and actual values
  predictions <- data.frame(matrix(ncol = ncol(hubs), nrow = 0))
  last_available <- data.frame(matrix(ncol = ncol(hubs), nrow = 0))
  actuals <- data.frame(matrix(ncol = ncol(hubs), nrow = 0))
  
  colnames(predictions) <- colnames(hubs)
  colnames(last_available) <- colnames(hubs)
  colnames(actuals) <- colnames(hubs)
  
  for (i in 1:validation_size) {

    train_size <- nrow(hubs) - test_size - validation_size - window_size + i
    hub_train <- hubs[1:train_size, ]

    start <- train_size - lags
    end <- train_size

    tvec <- TVECM(hub_train, include = "const", nthresh=nthresh,lag=lags, ngridBeta=100, ngridTh=150, plot=TRUE,trim=0.05, common="All")

    pred_data <- hubs[start:end, ]

    hub_forecast <- predict.TVECM(tvec, newdata=pred_data, n.ahead=window_size)
    
    hub_prediction <- hub_forecast[window_size, , drop = FALSE]

    hub_last_available <- hubs[end, ]
    hub_actual <- hubs[end + window_size, ]
    
    predictions <- rbind(predictions, hub_prediction)
    last_available <- rbind(last_available, hub_last_available)
    actuals <- rbind(actuals, hub_actual)
  }
  
  # Return both data frames as a list
  return(list(predictions = predictions, actuals = actuals, last_available = last_available))
}


tvecm_system <- function(hub1_name, hub2_name, validation_size = 250, test_size = 250, window_size = 5, nthresh = 1, lags=NULL, verbose = TRUE, save=TRUE) {

    hub_prices <- list(
        nbp = read.csv("../../data/interpolated/nbp_close_interpolated.csv"),
        peg = read.csv("../../data/interpolated/peg_close_interpolated.csv"),
        the = read.csv("../../data/interpolated/the_close_interpolated.csv"),
        ttf = read.csv("../../data/interpolated/ttf_close_interpolated.csv"),
        ztp = read.csv("../../data/interpolated/ztp_close_interpolated.csv")
    )

    hub1 <- hub_prices[[hub1_name]]
    hub2 <- hub_prices[[hub2_name]]

    hubs <- data.frame(hub1 = hub1$CLOSE, hub2 = hub2$CLOSE)

    train_size <- nrow(hubs) - test_size - window_size
    hub_train <- hubs[1:train_size + 1, ]
    if(is.null(lags)) {
      aics <- c()
      bics <- c()
      max_lag <- 8
      for (p in 1:max_lag) {
        tvecm <- TVECM(hub_train, include = "const", nthresh=nthresh,lag=p, ngridBeta=100, ngridTh=150, plot=TRUE,trim=0.05, common="All")
        aics <- c(aics, AIC(tvecm))
        bics <- c(bics, BIC(tvecm))
      }
      lags <- which.min(bics)
    }
      
    tvecm_output <- tvecm_test_predictions(hubs, window_size = window_size, test_size = test_size, nthresh = nthresh, lags = lags)
    tvecm_validation_output <- tvecm_validation_predictions(hubs, window_size = window_size, validation_size = validation_size, test_size = test_size, nthresh = nthresh, lags = lags)

    if (verbose) {

        print(paste0("Selected number of lags: ", lags))
        hub1_predictions <- tvecm_output$predictions$hub1
        hub1_actuals <- tvecm_output$actuals$hub1
        hub2_predictions <- tvecm_output$predictions$hub2
        hub2_actuals <- tvecm_output$actuals$hub2

        hub1_mae <- mae(hub1_actuals, hub1_predictions)
        hub2_mae <- mae(hub2_actuals, hub2_predictions)

        hub1_rmse <- rmse(hub1_actuals, hub1_predictions)
        hub2_rmse <- rmse(hub2_actuals, hub2_predictions)

        print(paste0("Pair: ", hub1_name, " | ", hub2_name))
        print(paste0("Window Size: ", window_size))
        print(paste0("Test Size: ", test_size))
        print(paste0(hub1_name,": Mean Absolute Error: ", round(hub1_mae, 3)))
        print(paste0(hub1_name,": Root Mean Squared Error: ", round(hub1_rmse, 3)))

        print(paste0(hub2_name,": Mean Absolute Error: ", round(hub2_mae, 3)))
        print(paste0(hub2_name,": Root Mean Squared Error: ", round(hub2_rmse, 3)))

    }

    if (save) {
        predictions <- tvecm_output$predictions
        last_available <- tvecm_output$last_available
        actuals <- tvecm_output$actuals
        colnames(predictions) <- c(hub1_name, hub2_name)
        colnames(last_available) <- c(hub1_name, hub2_name)
        colnames(actuals) <- c(hub1_name, hub2_name)
        prediction_dates <- tail(hub1$Date, test_size)
        predictions <- cbind(data.frame(Date = prediction_dates), predictions, row.names = NULL)
        last_available <- cbind(data.frame(Date = prediction_dates), last_available, row.names = NULL)
        actuals <- cbind(data.frame(Date = prediction_dates), actuals, row.names = NULL)

        write.csv(actuals, paste0("../../predictions/test/actuals/",hub1_name,"_", hub2_name, "_h", test_size, "_w", window_size, "_tvecm_long_", "t", nthresh, "_actuals.csv"), row.names = FALSE)
        write.csv(last_available, paste0("../../predictions/test/last_available/",hub1_name,"_", hub2_name, "_h", test_size, "_w", window_size, "_tvecm_long_", "t", nthresh, "_last_available.csv"), row.names = FALSE)
        write.csv(predictions, paste0("../../predictions/test/predictions/",hub1_name,"_", hub2_name, "_h", test_size, "_w", window_size, "_tvecm_long_", "t", nthresh, "_predictions.csv"), row.names = FALSE)

        predictions <- tvecm_validation_output$predictions
        last_available <- tvecm_validation_output$last_available
        actuals <- tvecm_validation_output$actuals
        colnames(predictions) <- c(hub1_name, hub2_name)
        colnames(last_available) <- c(hub1_name, hub2_name)
        colnames(actuals) <- c(hub1_name, hub2_name)
        prediction_dates <- tail(hub1$Date, test_size + validation_size)
        predictions <- cbind(data.frame(Date = prediction_dates), predictions, row.names = NULL)
        last_available <- cbind(data.frame(Date = prediction_dates), last_available, row.names = NULL)
        actuals <- cbind(data.frame(Date = prediction_dates), actuals, row.names = NULL)
        write.csv(actuals, paste0("../../predictions/validation/actuals/",hub1_name,"_", hub2_name,"_v", validation_size, "_h", test_size, "_w", window_size, "_tvecm_long_", "t", nthresh, "_actuals.csv"), row.names = FALSE)
        write.csv(last_available, paste0("../../predictions/validation/last_available/",hub1_name,"_", hub2_name, "_v", validation_size, "_h", test_size, "_w", window_size,  "_tvecm_long_", "t", nthresh, "_last_available.csv"), row.names = FALSE)
        write.csv(predictions, paste0("../../predictions/validation/predictions/",hub1_name,"_", hub2_name, "_v", validation_size, "_h", test_size, "_w", window_size,  "_tvecm_long_", "t", nthresh, "_predictions.csv"), row.names = FALSE)
    }

    return(tvecm_output)

}

tvecm_training_model <- function(hub1_name, hub2_name, test_size = 250, window_size = 5, nthresh=1, lags=NULL) {

    hub_prices <- list(
        nbp = read.csv("../../data/interpolated/nbp_close_interpolated.csv"),
        peg = read.csv("../../data/interpolated/peg_close_interpolated.csv"),
        the = read.csv("../../data/interpolated/the_close_interpolated.csv"),
        ttf = read.csv("../../data/interpolated/ttf_close_interpolated.csv"),
        ztp = read.csv("../../data/interpolated/ztp_close_interpolated.csv")
    )

    hub1 <- hub_prices[[hub1_name]]
    hub2 <- hub_prices[[hub2_name]]

    hubs <- data.frame(hub1 = hub1$CLOSE, hub2 = hub2$CLOSE)

    train_size <- nrow(hubs) - test_size - window_size
    hub_train <- hubs[1:train_size + 1, ]

    if(is.null(lags)) {
       aics <- c()
      bics <- c()
      max_lag <- 8
      for (p in 1:max_lag) {
        tvecm <- TVECM(hub_train, include = "const", nthresh=nthresh,lag=p, ngridBeta=100, ngridTh=150, plot=TRUE,trim=0.05, common="All")
        aics <- c(aics, AIC(tvecm))
        bics <- c(bics, BIC(tvecm))
      }
      lags <- which.min(bics)
    }

    vecm_model <- TVECM(hub_train, include = "const", nthresh=nthresh,lag=lags, ngridBeta=100, ngridTh=150, plot=TRUE,trim=0.05, common="All")

    return(vecm_model)


}
