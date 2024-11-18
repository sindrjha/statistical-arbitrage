if (require("rugarch") == FALSE) {
  install.packages("rugarch")
  library(rugarch)
}

cointegration <- function(hub1_name, hub2_name, rolling_window, validation_size = 250, test_size = 250, window_size = 5, model = "ols", garch_model = "sGARCH", garch_order = c(1,1), garch_dist="norm", save=TRUE) {

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

    results <- data.frame(
        Date = hub1$Date,
        hub1 = hub1$CLOSE,
        hub2 = hub2$CLOSE,
        alpha = rep(NA, nrow(hubs)),
        beta = rep(NA, nrow(hubs)),
        sigma = rep(NA, nrow(hubs)),
        residuals = rep(NA, nrow(hubs))  # Initialize with NA
    )

    start <- nrow(hubs) - validation_size - test_size - window_size


    for (i in start :nrow(hubs)) {
        hub_train <- hubs[(i - rolling_window+1):i, ]

        if (model == "ols") {
            ols <- lm(hub_train$hub1 ~ hub_train$hub2)
            alpha <- coef(ols)[1]
            beta <- coef(ols)[2]

        }



        resids <- residuals(ols)


        spec <- ugarchspec(
        variance.model = list(model = garch_model, garchOrder = garch_order),
        mean.model = list(armaOrder = c(0,0), include.mean = FALSE),
        distribution.model = garch_dist
        )
        garch_fit <- tryCatch({
            ugarchfit(spec = spec, data = resids, solver = "hybrid")
        }, error = function(e) {
            message("Hybrid solver failed, trying default solver.")
            ugarchfit(spec = spec, data = resids)
        })
        sigma <- tail(sigma(garch_fit), 1)
        
        results$alpha[i] <- alpha
        results$beta[i] <- beta
        results$sigma[i] <- sigma
        results$residuals[i] <- hubs$hub1[i] - alpha - beta * hubs$hub2[i]
    }

    colnames(results) <- c("Date", hub1_name, hub2_name, "alpha", "beta", "Sigma", "residuals")

    if (save) {
        write.csv(results, paste0("../../predictions/cointegration/",hub1_name,"_", hub2_name, "_r", rolling_window, "_v", validation_size, "_h", test_size, "_w", window_size, "_", model, "_cointegration.csv"), row.names = FALSE)
        validation_results <- results[(nrow(results) - test_size - validation_size - window_size + 1):(nrow(results) - test_size  - window_size), ]
        test_results <- results[(nrow(results) - test_size - window_size + 1):(nrow(results) - window_size), ]
        if (all(garch_order == c(1, 1))) {
          write.csv(validation_results, paste0("../../predictions/validation/predictions/",hub1_name,"_", hub2_name, "_v", validation_size, "_h", test_size, "_w", window_size,  "_", garch_model, "_",  garch_dist, "_cointegration_predictions.csv"), row.names = FALSE)
          write.csv(test_results, paste0("../../predictions/test/predictions/",hub1_name,"_", hub2_name, "_h", test_size, "_w", window_size,  "_", garch_model, "_",  garch_dist, "_cointegration_predictions.csv"), row.names = FALSE)
        } else {
          write.csv(validation_results, paste0("../../predictions/validation/predictions/",hub1_name,"_", hub2_name, "_v", validation_size, "_h", test_size, "_w", window_size,  "_", garch_model, "_",  garch_dist, "_", garch_order[1], garch_order[2], "_cointegration_predictions.csv"), row.names = FALSE)
          write.csv(test_results, paste0("../../predictions/test/predictions/",hub1_name,"_", hub2_name, "_h", test_size, "_w", window_size, "_", garch_model, "_",  garch_dist, "_", garch_order[1], garch_order[2], "_cointegration_predictions.csv"), row.names = FALSE)
        }
    }

    return(results)

}

cointegration_volatilities <- function(results, hub1_name, hub2_name, model = "sGARCH", dist = "norm", garch_order = c(1,1), validation_size = 250, test_size = 250, window_size = 5, save=TRUE) {

  garch_validation_predictions <- garch_predictions(results, model = model, dist = dist, garch_order =  garch_order, window_size = window_size, validation_size = validation_size, test_size = test_size, mode = "validation")
  garch_predictions <- garch_predictions(results, model = model, dist = dist, garch_order =  garch_order, window_size = window_size, validation_size = validation_size, test_size = test_size, mode = "test")

  if (save) {
    validation_prediction_dates <- head(tail(results$Date, test_size + validation_size), validation_size)
    garch_validation_predictions <- cbind(data.frame(Date = validation_prediction_dates), garch_validation_predictions, row.names = NULL)
    write.csv(garch_validation_predictions, paste0("../../predictions/validation/predictions/",hub1_name,"_", hub2_name, "_v", validation_size, "_h", test_size, "_w", window_size, "_", model, "_", dist, "_", garch_order[1], garch_order[2], "_cointegration_predictions.csv"), row.names = FALSE)

    test_prediction_dates <- tail(results$Date, test_size)
    garch_predictions <- cbind(data.frame(Date = test_prediction_dates), garch_predictions, row.names = NULL)
    write.csv(garch_predictions, paste0("../../predictions/test/predictions/",hub1_name,"_", hub2_name, "_h", test_size, "_w", window_size,  "_", model, "_", dist, "_", garch_order[1], garch_order[2], "_cointegration_predictions.csv"), row.names = FALSE)
  }


  # Return the predictions
  return(list(predictions = garch_predictions, validation_predictions = garch_validation_predictions))
}


garch_predictions <- function(results, model = "sGARCH", dist = "norm", garch_order =c(1,1), window_size = 5, validation_size = 250, test_size = 250, mode = "validation") {
  # Initialize empty data frames for storing predictions and actual values
  difference <- results$hub_diff
  predictions <- data.frame(matrix(ncol = 1, nrow = 0))
  colnames(predictions) <- c("Sigma")

  difference <- difference[!is.na(difference)]


  if (mode == "validation") {
    size <- validation_size
    train_size_base <- length(difference) - test_size - validation_size - window_size
  } else {
    size <- test_size
    train_size_base <- length(difference) - test_size - window_size
  }
  

  for (i in 1:size) {
    train_size <- train_size_base + i
    difference_train <- difference[1:train_size]

    #arma_model <- auto.arima(difference_train, seasonal = FALSE)

    spec <- ugarchspec(
    variance.model = list(model = model, garchOrder = garch_order),
    mean.model = list(armaOrder = c(0,0), include.mean = FALSE),
    distribution.model = dist
    )
    garch_fit <- tryCatch({
        ugarchfit(spec = spec, data = difference_train, solver = "hybrid")
    }, error = function(e) {
        message("Hybrid solver failed, trying default solver.")
        ugarchfit(spec = spec, data = difference_train)
    })
    garch_prediction <- tail(sigma(garch_fit), 1)

    colnames(garch_prediction) <- c("Sigma")

    predictions <- rbind(predictions, garch_prediction)
    }
    return(predictions)
}
