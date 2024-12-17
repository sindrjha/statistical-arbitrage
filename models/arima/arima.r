if (require("dplyr") == FALSE) {
  install.packages("dplyr")
  library(dplyr)
}
if (require("zoo") == FALSE) {
  install.packages("zoo")
  library(zoo)
}
if (require("psych") == FALSE) {
  install.packages("psych")
  library(psych)
}
if (require("TSA") == FALSE) {
  install.packages("TSA")
  library(TSA)
}
if (require("forecast") == FALSE) {
  install.packages("forecast")
  library(forecast)
}
if (require("Metrics") == FALSE) {
  install.packages("Metrics")
  library(Metrics)
}



arima_test_predictions <- function(hub,  test_size = 250, window_size = 5, arima_order) {
  
  predictions <- data.frame(matrix(ncol = ncol(hub), nrow = 0))
  last_available <- data.frame(matrix(ncol = ncol(hub), nrow = 0))
  actuals <- data.frame(matrix(ncol = ncol(hub), nrow = 0))
  
  names(predictions) <- names(hub)
  names(last_available) <- names(hub)
  names(actuals) <- names(hub)
  
  for (i in 1:test_size) {
    train_size <- nrow(hub) - test_size - window_size + i
    hub_train <- hub[1:train_size, ]
    
    arima_model <- arima(hub_train, order = arima_order)
    
    hub_forecast <- predict(arima_model, n.ahead = window_size)$pred
    
    hub_prediction <- hub_forecast[window_size]
    
    prediction_data <- data.frame(hub_prediction)
    names(prediction_data) <- names(hub)

    hub_last_available <- hub[train_size, ]
    names(hub_last_available) <- names(hub)
    
    hub_actual <- hub[train_size + window_size, ]
    actual_data <- data.frame(hub_actual)
    names(actual_data) <- names(hub)
    
    predictions <- rbind(predictions, prediction_data)
    last_available <- rbind(last_available, hub_last_available)
    actuals <- rbind(actuals, actual_data)
  }
  
  return(list(predictions = predictions, actuals = actuals, last_available = last_available))
}

arima_validation_predictions <- function(hub, validation_size=250, test_size = 250, window_size = 5, arima_order) {
  predictions <- data.frame(matrix(ncol = ncol(hub), nrow = 0))
  last_available <- data.frame(matrix(ncol = ncol(hub), nrow = 0))
  actuals <- data.frame(matrix(ncol = ncol(hub), nrow = 0))
  
  names(predictions) <- names(hub)
  names(last_available) <- names(hub)
  names(actuals) <- names(hub)
  
  for (i in 1:validation_size) {
    train_size <- nrow(hub) - test_size - validation_size - window_size + i
    hub_train <- hub[1:train_size, ]
    
    arima_model <- arima(hub_train, order = arima_order)
    
    hub_forecast <- predict(arima_model, n.ahead = window_size)$pred
    
    hub_prediction <- hub_forecast[window_size]
    
    prediction_data <- data.frame(hub_prediction)
    names(prediction_data) <- names(hub)
    
    hub_last_available <- hub[train_size, ]
    names(hub_last_available) <- names(hub)
    
    hub_actual <- hub[train_size + window_size, ]
    actual_data <- data.frame(hub_actual)
    names(actual_data) <- names(hub)
    
    predictions <- rbind(predictions, prediction_data)
    last_available <- rbind(last_available, hub_last_available)
    actuals <- rbind(actuals, actual_data)
  }
  
  return(list(predictions = predictions, actuals = actuals, last_available = last_available))
}


arima_system <- function(hub1_name, validation_size=250, test_size = 250, window_size = 5, verbose = TRUE, save = TRUE) {


  hub_prices <- list(
    nbp = read.csv("../../data/interpolated/nbp_close_interpolated.csv"),
    peg = read.csv("../../data/interpolated/peg_close_interpolated.csv"),
    the = read.csv("../../data/interpolated/the_close_interpolated.csv"),
    ttf = read.csv("../../data/interpolated/ttf_close_interpolated.csv"),
    ztp = read.csv("../../data/interpolated/ztp_close_interpolated.csv")
  )

  hub1 <- hub_prices[[hub1_name]]
  hub <- data.frame(hub = hub1$CLOSE)

  train_size <- nrow(hub) - test_size - window_size + 1
  hub_train <- hub[1:train_size, ]

  arima_model <- auto.arima(hub_train, max.p = 20, max.q = 20, ic = "aic")

  arima_order <- arima_model$arma[c(1, 6, 2)]

  arima_output <- arima_test_predictions(hub, test_size = test_size, window_size = window_size, arima_order = arima_order)
  arima_validation_output <- arima_validation_predictions(hub, validation_size = validation_size, test_size = test_size, window_size = window_size, arima_order = arima_order)

  if (verbose) {
    print(paste("AR:", arima_order[1], "I:", arima_order[2], "MA:", arima_order[3]))
    hub_predictions <- arima_output$predictions$hub
    hub_actuals <- arima_output$actuals$hub

    hub_mae <- mae(hub_actuals, hub_predictions)
    hub_rmse <- rmse(hub_actuals, hub_predictions)

    print(paste0(hub1_name, ": Mean Absolute Error: ", hub_mae))
    print(paste0(hub1_name, ": Root Mean Squared Error: ", hub_rmse))
  }

  if (save) {
    predictions <- arima_output$predictions
    last_available <- arima_output$last_available
    actuals <- arima_output$actuals
    names(predictions) <- c(hub1_name)
    names(last_available) <- c(hub1_name)
    names(actuals) <- c(hub1_name)
    prediction_dates <- tail(hub1$Date, test_size)
    predictions <- cbind(data.frame(Date = prediction_dates), predictions, row.names = NULL)
    last_available <- cbind(data.frame(Date = prediction_dates), last_available, row.names = NULL)
    actuals <- cbind(data.frame(Date = prediction_dates), actuals, row.names = NULL)

    write.csv(actuals, paste0("../../predictions/test/actuals/", hub1_name, "_h", test_size, "_w", window_size, "_arima_actuals.csv"), row.names = FALSE)
    write.csv(last_available, paste0("../../predictions/test/last_available/", hub1_name, "_h", test_size, "_w", window_size, "_arima_last_available.csv"), row.names = FALSE)
    write.csv(predictions, paste0("../../predictions/test/predictions/", hub1_name, "_h", test_size, "_w", window_size, "_arima_predictions.csv"), row.names = FALSE)


    predictions <- arima_validation_output$predictions
    last_available <- arima_validation_output$last_available
    actuals <- arima_validation_output$actuals
    names(predictions) <- c(hub1_name)
    names(last_available) <- c(hub1_name)
    names(actuals) <- c(hub1_name)
    prediction_dates <- head(tail(hub1$Date, test_size + validation_size), validation_size)
    predictions <- cbind(data.frame(Date = prediction_dates), predictions, row.names = NULL)
    last_available <- cbind(data.frame(Date = prediction_dates), last_available, row.names = NULL)
    actuals <- cbind(data.frame(Date = prediction_dates), actuals, row.names = NULL)
    write.csv(actuals, paste0("../../predictions/validation/actuals/", hub1_name, "_v", validation_size, "_h", test_size, "_w", window_size, "_arima_actuals.csv"), row.names = FALSE)
    write.csv(last_available, paste0("../../predictions/validation/last_available/", hub1_name, "_v", validation_size, "_h", test_size, "_w", window_size, "_arima_last_available.csv"), row.names = FALSE)
    write.csv(predictions, paste0("../../predictions/validation/predictions/", hub1_name, "_v", validation_size, "_h", test_size, "_w", window_size, "_arima_predictions.csv"), row.names = FALSE)

  }

  return(arima_output)
}

arima_training_model <- function(hub1_name, test_size = 250, window_size = 5, arima_order) {

  hub_prices <- list(
    nbp = read.csv("../../data/interpolated/nbp_close_interpolated.csv"),
    peg = read.csv("../../data/interpolated/peg_close_interpolated.csv"),
    the = read.csv("../../data/interpolated/the_close_interpolated.csv"),
    ttf = read.csv("../../data/interpolated/ttf_close_interpolated.csv"),
    ztp = read.csv("../../data/interpolated/ztp_close_interpolated.csv")
  )

  hub1 <- hub_prices[[hub1_name]]
  hub <- data.frame(hub = hub1$CLOSE)

  train_size <- nrow(hub) - test_size - window_size + 1
  hub_train <- hub[1:train_size, ]

  arima_model <- auto.arima(hub_train, max.p = 20, max.q = 20, ic = "aic")

  arima_order <- arima_model$arma[c(1, 6, 2)]

  arima_output <- arima_test_predictions(hub, test_size = test_size, window_size = window_size, arima_order = arima_order)

  return(arima_output)
}
