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

vecm_test_predictions <- function(hubs, window_size = 5, test_size = 250, lags) {
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
    # Fit the VECM model
    vecm <- VECM(hub_train, lag = lags, r = 1, include = "const", estim = "ML")
    
    # Predict the future values
    hub_forecast <- predict(vecm, n.ahead = window_size)
    
    hub_prediction <- hub_forecast[window_size, , drop = FALSE]

    hub_last_available <- hubs[train_size, ]
    hub_actual <- hubs[train_size + window_size, ]
    
    predictions <- rbind(predictions, hub_prediction)
    last_available <- rbind(last_available, hub_last_available)
    actuals <- rbind(actuals, hub_actual)
  }
  
  # Return both data frames as a list
  return(list(predictions = predictions, actuals = actuals, last_available = last_available))
}

vecm_validation_predictions <- function(hubs, window_size = 5, validation_size = 250, test_size = 250, lags) {
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
    # Fit the VECM model
    vecm <- VECM(hub_train, lag = lags, r = 1, include = "const", estim = "ML")
    
    # Predict the future values
    hub_forecast <- predict(vecm, n.ahead = window_size)
    
    hub_prediction <- hub_forecast[window_size, , drop = FALSE]
    
    hub_last_available <- hubs[train_size, ]
    hub_actual <- hubs[train_size + window_size, ]
    
    predictions <- rbind(predictions, hub_prediction)
    last_available <- rbind(last_available, hub_last_available)
    actuals <- rbind(actuals, hub_actual)
  }
  
  # Return both data frames as a list
  return(list(predictions = predictions, actuals = actuals, last_available = last_available))
}


vecm_system <- function(hub1_name, hub2_name, validation_size = 250, test_size = 250, window_size = 5, lags=NULL, verbose = TRUE, save=TRUE) {

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
        max_lag <- 20
        for (p in 1:max_lag) {
        vecm <- VECM(hub_train, lag = p,  r = 1, include = "const", estim = "ML")
        aics <- c(aics, AIC(vecm))
        bics <- c(bics, BIC(vecm))
        }
        lags <- which.min(bics)
        
    }

    vecm_output <- vecm_test_predictions(hubs, window_size = window_size, test_size = test_size, lags = lags)
    vecm_validation_output <- vecm_validation_predictions(hubs, window_size = window_size, validation_size = validation_size, test_size = test_size, lags = lags)

    if (verbose) {

        print(paste0("Selected number of lags: ", lags))
        hub1_predictions <- vecm_output$predictions$hub1
        hub1_actuals <- vecm_output$actuals$hub1
        hub2_predictions <- vecm_output$predictions$hub2
        hub2_actuals <- vecm_output$actuals$hub2

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
        predictions <- vecm_output$predictions
        last_available <- vecm_output$last_available
        actuals <- vecm_output$actuals
        colnames(predictions) <- c(hub1_name, hub2_name)
        colnames(last_available) <- c(hub1_name, hub2_name)
        colnames(actuals) <- c(hub1_name, hub2_name)
        prediction_dates <- tail(hub1$Date, test_size)
        predictions <- cbind(data.frame(Date = prediction_dates), predictions, row.names = NULL)
        last_available <- cbind(data.frame(Date = prediction_dates), last_available, row.names = NULL)
        actuals <- cbind(data.frame(Date = prediction_dates), actuals, row.names = NULL)

        write.csv(actuals, paste0("../../predictions/test/actuals/",hub1_name,"_", hub2_name, "_h", test_size, "_w", window_size, "_vecm_actuals.csv"), row.names = FALSE)
        write.csv(last_available, paste0("../../predictions/test/last_available/",hub1_name,"_", hub2_name, "_h", test_size, "_w", window_size, "_vecm_last_available.csv"), row.names = FALSE)
        write.csv(predictions, paste0("../../predictions/test/predictions/",hub1_name,"_", hub2_name, "_h", test_size, "_w", window_size, "_vecm_predictions.csv"), row.names = FALSE)

        predictions <- vecm_validation_output$predictions
        last_available <- vecm_validation_output$last_available
        actuals <- vecm_validation_output$actuals
        colnames(predictions) <- c(hub1_name, hub2_name)
        colnames(last_available) <- c(hub1_name, hub2_name)
        colnames(actuals) <- c(hub1_name, hub2_name)
        prediction_dates <- tail(hub1$Date, test_size + validation_size)
        predictions <- cbind(data.frame(Date = prediction_dates), predictions, row.names = NULL)
        last_available <- cbind(data.frame(Date = prediction_dates), last_available, row.names = NULL)
        actuals <- cbind(data.frame(Date = prediction_dates), actuals, row.names = NULL)
        write.csv(actuals, paste0("../../predictions/validation/actuals/",hub1_name,"_", hub2_name,"_v", validation_size, "_h", test_size, "_w", window_size, "_vecm_actuals.csv"), row.names = FALSE)
        write.csv(last_available, paste0("../../predictions/validation/last_available/",hub1_name,"_", hub2_name, "_v", validation_size, "_h", test_size, "_w", window_size, "_vecm_last_available.csv"), row.names = FALSE)
        write.csv(predictions, paste0("../../predictions/validation/predictions/",hub1_name,"_", hub2_name, "_v", validation_size, "_h", test_size, "_w", window_size, "_vecm_predictions.csv"), row.names = FALSE)
    }

    return (vecm_output)

}

vecm_training_model <- function(hub1_name, hub2_name, test_size = 250, window_size = 5, lags=NULL) {

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
        max_lag <- 20
        for (p in 1:max_lag) {
            vecm <- VECM(hub_train, lag = p,  r = 1, include = "const", estim = "ML")
            aics <- c(aics, AIC(vecm))
            bics <- c(bics, BIC(vecm))
        }
        lags <- which.min(bics)
    }

    vecm_model <- VECM(hub_train, lag = lags, r = 1, include = "const", estim = "ML")

    return(vecm_model)


}
