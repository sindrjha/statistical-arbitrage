# Install required packages
if (require("dplyr") == FALSE) {
  install.packages("dplyr")
  library(dplyr)
}
if (require("zoo") == FALSE) {
  install.packages("zoo")
  library(zoo)
}

if (require("forecast") == FALSE) {
  install.packages("forecast")
  library(forecast)
}
if (require("tseries") == FALSE) {
  install.packages("tseries")
  library(tseries)
}
if (require("moments") == FALSE) {
  install.packages("moments")
  library(moments)
}
if (require("tibble") == FALSE) {
  install.packages("tibble")
  library(tibble)
}
if (require("urca") == FALSE) {
  install.packages("urca")
  library(urca)
}
if (require("kableExtra") == FALSE) {
  install.packages("kableExtra")
  library(kableExtra)
}
if (require("rugarch") == FALSE) {
  install.packages("rugarch")
  library(rugarch)
}

garch_test_predictions <- function(hubs, model = "sGARCH", dist = "norm", window_size = 5, test_size = 250, garch_order = c(1, 1)) {
  # Initialize empty data frames for storing predictions and actual values
    predictions <- data.frame(matrix(ncol = 1, nrow = 0))
    colnames(predictions) <- c("Sigma")

    start <- 1
    end <- nrow(hubs) - window_size
    return_start <- window_size + 1
    return_end <- nrow(hubs)

    hub1_historical_returns <- log(hubs[return_start:return_end, 1] / hubs[start:end, 1])
    hub2_historical_returns <- log(hubs[return_start:return_end, 2] / hubs[start:end, 2])

    returns_difference <- hub1_historical_returns - hub2_historical_returns


  for (i in 1:test_size) {
    train_size <- length(returns_difference) - test_size - window_size + i
    returns_difference_train <- returns_difference[1:train_size]

    #arma_model <- auto.arima(returns_difference_train, seasonal = FALSE)


    spec <- ugarchspec(
    variance.model = list(model = model, garchOrder = garch_order),
    #mean.model = list(armaOrder = c(arma_model$arma[1], arma_model$arma[2]), include.mean = TRUE),
    mean.model = list(armaOrder = c(0,0), include.mean = FALSE),
    distribution.model = dist
    )

    # Step 3: Fit the GARCH model
    garch_fit <- tryCatch({
        ugarchfit(spec = spec, data = returns_difference_train, solver = "hybrid")
    }, error = function(e) {
        message("Hybrid solver failed, trying default solver.")
        ugarchfit(spec = spec, data = returns_difference_train)
    })

    garch_forecast <- ugarchforecast(garch_fit, n.ahead = window_size)

    garch_prediction <- garch_forecast@forecast$sigmaFor[window_size, , drop = FALSE]

    colnames(garch_prediction) <- c("Sigma")

    predictions <- rbind(predictions, garch_prediction)
    }
    return(predictions)
}

garch_validation_predictions <- function(hubs, model = "sGARCH", dist = "norm", window_size = 5, validation_size = 250, test_size = 250, garch_order = c(1, 1)) {
  # Initialize empty data frames for storing predictions and actual values
    predictions <- data.frame(matrix(ncol = 1, nrow = 0))
    colnames(predictions) <- c("Sigma")

    start <- 1
    end <- nrow(hubs) - window_size
    return_start <- window_size + 1
    return_end <- nrow(hubs)

    hub1_historical_returns <- log(hubs[return_start:return_end, 1] / hubs[start:end, 1])
    hub2_historical_returns <- log(hubs[return_start:return_end, 2] / hubs[start:end, 2])

    returns_difference <- hub1_historical_returns - hub2_historical_returns

  for (i in 1:validation_size) {
    train_size <- length(returns_difference) - test_size - validation_size - window_size + i
    returns_difference_train <- returns_difference[1:train_size]

    #arma_model <- auto.arima(returns_difference_train, seasonal = FALSE)

    spec <- ugarchspec(
    variance.model = list(model = model, garchOrder = garch_order),
    #mean.model = list(armaOrder = c(arma_model$arma[1], arma_model$arma[2]), include.mean = TRUE),
    mean.model = list(armaOrder = c(0,0), include.mean = FALSE),
    distribution.model = dist
    )
    garch_fit <- tryCatch({
        ugarchfit(spec = spec, data = returns_difference_train, solver = "hybrid")
    }, error = function(e) {
        message("Hybrid solver failed, trying default solver.")
        ugarchfit(spec = spec, data = returns_difference_train)
    })
    garch_forecast <- ugarchforecast(garch_fit, n.ahead = window_size)
    garch_prediction <- garch_forecast@forecast$sigmaFor[window_size, , drop = FALSE]

    colnames(garch_prediction) <- c("Sigma")

    predictions <- rbind(predictions, garch_prediction)
    }
    return(predictions)
}

garch_system <- function(hub1_name, hub2_name, model = "sGARCH", dist = "norm", validation_size = 250, test_size = 250, window_size = 5, garch_order=c(1,1), verbose = TRUE, save=TRUE) {
  # Load the data
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

  # Perform the GARCH model
  garch_validation_predictions <- garch_validation_predictions(hubs, model = model, dist = dist, window_size = window_size, validation_size = validation_size, test_size = test_size, garch_order = garch_order)
  garch_predictions <- garch_test_predictions(hubs, model = model, dist = dist, window_size = window_size, test_size = test_size, garch_order = garch_order)
  

  if (save) {
    validation_prediction_dates <- head(tail(hub1$Date, test_size + validation_size), validation_size)
    garch_validation_predictions <- cbind(data.frame(Date = validation_prediction_dates), garch_validation_predictions, row.names = NULL)
    write.csv(garch_validation_predictions, paste0("../../predictions/validation/predictions/",hub1_name,"_", hub2_name, "_v", validation_size, "_h", test_size, "_w", window_size, "_", model, "_", dist, "_",garch_order[1],garch_order[2],"_predictions.csv"), row.names = FALSE)

    test_prediction_dates <- tail(hub1$Date, test_size)
    garch_predictions <- cbind(data.frame(Date = test_prediction_dates), garch_predictions, row.names = NULL)
    write.csv(garch_predictions, paste0("../../predictions/test/predictions/",hub1_name,"_", hub2_name, "_h", test_size, "_w", window_size,  "_", model, "_", dist, "_",garch_order[1],garch_order[2],"_predictions.csv"), row.names = FALSE)
  }


  # Return the predictions
  return(list(predictions = garch_predictions, validation_predictions = garch_validation_predictions))
}