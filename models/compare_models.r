if (require("dplyr") == FALSE) {
  install.packages("dplyr")
  library(dplyr)
}
if (require("zoo") == FALSE) {
  install.packages("zoo")
  library(zoo)
}
if (require("pander") == FALSE) {
  install.packages("pander")
  library(pander)
}
if (require("Metrics") == FALSE) {
  install.packages("Metrics")
  library(Metrics)
}
if (require("forecast") == FALSE) {
  install.packages("forecast")
  library(forecast)
}

directional_accuracy <- function(actual, predicted) {
  correct_directions <- sum((sign(actual) == sign(predicted)))
  da <- correct_directions / length(actual)
  
  return(da * 100) 
}

mean_directional_loss <- function(actual, predicted) {
  abs_diff <- abs(actual - predicted)
  
  directional_mismatch <- sign(actual) != sign(predicted)
  
  mdl <- mean(directional_mismatch * abs_diff)
  
  return(mdl)
}

profit_weighted_error <- function(actual, predicted) {
  abs_diff <- abs(actual - predicted)
  
  profit_impact <- actual * sign(predicted)
  
  weighted_errors <- sum(abs_diff * profit_impact)
  
  total_abs_actual <- sum(abs(actual))
  
  pwe <- weighted_errors / total_abs_actual
  
  return(pwe)
}
directional_value <- function(actual, predicted) {
  correct_direction <- sign(actual) == sign(predicted)
  wrong_direction <- sign(actual) != sign(predicted)
  
  directional_value_result <- (sum(abs(actual) * correct_direction) - sum(abs(actual) * wrong_direction)) / length(actual)
  
  return(directional_value_result)
}


compare_error_metrics <- function(hub_pairs, models, horizon, window_size) {


    error_metrics <- c("mape", "rmse", "rmse_return", "da_return", "dv_return", "rmse_spread", "da_spread", "corr_spread", "dv_spread")


    error_metrics_results <- list()

    for (pair in hub_pairs) {
    hub1_name <- pair[1]
    hub2_name <- pair[2]
    
    hub1_df <- data.frame(matrix(NA, nrow = length(models), ncol = length(error_metrics)))
    colnames(hub1_df) <- error_metrics
    rownames(hub1_df) <- models
    
    hub2_df <- hub1_df
    
    for (i in 1:length(models)) {
        model1 <- models[i]
        if (grepl("arima", model1) || grepl("naive", model1)) {
                model1_hub1_filename <- paste0("../predictions/test/predictions/", hub1_name, "_h", horizon, "_w", window_size, "_", model1, "_predictions.csv")
                model1_hub2_filename <- paste0("../predictions/test/predictions/", hub2_name, "_h", horizon, "_w", window_size, "_", model1, "_predictions.csv")
                model1_hub1_predictions <- read.csv(model1_hub1_filename)
                model1_hub2_predictions <- read.csv(model1_hub2_filename)
                model1_predictions <- cbind(model1_hub1_predictions, model1_hub2_predictions)
            }
        else {
            model1_filename <- paste0("../predictions/test/predictions/", hub1_name, "_", hub2_name, "_h", horizon, "_w", window_size, "_", model1, "_predictions.csv")
            model1_predictions <- read.csv(model1_filename)
        }
        
        actuals_hub1_filename <- paste0("../data/interpolated/", hub1_name, "_close_interpolated.csv")
        hub1_historical_data <- read.csv(actuals_hub1_filename)
        colnames(hub1_historical_data) <- c("Date", hub1_name)
        actuals_hub1 <- tail(hub1_historical_data, horizon)
        
            
        actuals_hub2_filename <- paste0("../data/interpolated/", hub2_name, "_close_interpolated.csv")
        hub2_historical_data <- read.csv(actuals_hub2_filename)
        colnames(hub2_historical_data) <- c("Date", hub2_name)
        actuals_hub2 <- tail(hub2_historical_data, horizon)

        

        hub1_actual_returns <- log(actuals_hub1[[hub1_name]] / tail(lag(hub1_historical_data[[hub1_name]], window_size), horizon)) *100
        hub2_actual_returns <- log(actuals_hub2[[hub2_name]] / tail(lag(hub2_historical_data[[hub2_name]], window_size), horizon)) *100

        hub1_predicted_returns <- log(model1_predictions[[hub1_name]] / tail(lag(hub1_historical_data[[hub1_name]], window_size), horizon)) *100
        hub2_predicted_returns <- log(model1_predictions[[hub2_name]] / tail(lag(hub2_historical_data[[hub2_name]], window_size), horizon)) *100

        actual_spread <- hub1_actual_returns - hub2_actual_returns
        predicted_spread <- hub1_predicted_returns - hub2_predicted_returns


        
        model_mape_hub1 <- mape(actuals_hub1[[hub1_name]], model1_predictions[[hub1_name]])*100
        model_rmse_hub1 <- rmse(actuals_hub1[[hub1_name]], model1_predictions[[hub1_name]])

        model_mape_hub2 <- mape(actuals_hub2[[hub2_name]], model1_predictions[[hub2_name]])*100
        model_rmse_hub2 <- rmse(actuals_hub2[[hub2_name]], model1_predictions[[hub2_name]])

        rmse_return_hub1 <- rmse(hub1_actual_returns, hub1_predicted_returns)
        rmse_return_hub2 <- rmse(hub2_actual_returns, hub2_predicted_returns)

        rmse_spread <- rmse(actual_spread, predicted_spread)
        da_spread <- directional_accuracy(actual_spread, predicted_spread)
        corr_spread <- cor(actual_spread, predicted_spread, method = "pearson")
        dv_spread <- directional_value(actual_spread, predicted_spread)


        da_return_hub1 <- directional_accuracy(hub1_actual_returns, hub1_predicted_returns)
        da_return_hub2 <- directional_accuracy(hub2_actual_returns, hub2_predicted_returns)
        dv_return_hub1 <- directional_value(hub1_actual_returns, hub1_predicted_returns)
        dv_return_hub2 <- directional_value(hub2_actual_returns, hub2_predicted_returns)

        
        hub1_df[i, "mape"] <- round(model_mape_hub1, 3)
        hub1_df[i, "rmse"] <- round(model_rmse_hub1, 3)
        hub1_df[i, "rmse_return"] <- round(rmse_return_hub1, 3)
        hub1_df[i, "da_return"] <- round(da_return_hub1, 3)
        hub1_df[i, "dv_return"] <- round(dv_return_hub1, 3)
        hub1_df[i, "rmse_spread"] <- round(rmse_spread, 3)
        hub1_df[i, "da_spread"] <- round(da_spread, 3)
        hub1_df[i, "corr_spread"] <- round(corr_spread, 3)
        hub1_df[i, "dv_spread"] <- round(dv_spread, 3)


        
        hub2_df[i, "mape"] <- round(model_mape_hub2, 3)
        hub2_df[i, "rmse"] <- round(model_rmse_hub2, 3)
        hub2_df[i, "rmse_return"] <- round(rmse_return_hub2, 3)
        hub2_df[i, "da_return"] <- round(da_return_hub2, 3)
        hub2_df[i, "dv_return"] <- round(dv_return_hub2, 3)
        hub2_df[i, "rmse_spread"] <- round(rmse_spread, 3)
        hub2_df[i, "da_spread"] <- round(da_spread, 3)
        hub2_df[i, "corr_spread"] <- round(corr_spread, 3)
    }

    hub_results <- list(hub1 = hub1_df, hub2 = hub2_df)
    names(hub_results) <- c(hub1_name, hub2_name)
    error_metrics_results[[paste(hub1_name, hub2_name, sep = "_")]] <- hub_results
    }

    add_star_to_min_post_calculation <- function(df) {
    for (metric in colnames(df)) {
        min_value <- min(as.numeric(df[[metric]]), na.rm = TRUE)
        df[[metric]] <- sapply(df[[metric]], function(x) {
        numeric_x <- as.numeric(x)
        if (!is.na(numeric_x) && numeric_x == min_value) {
            paste0(format(numeric_x, digits = 3), "~")
        } else {
            paste0(format(numeric_x, digits = 3), " ")
        }
        })
    }
    return(df)
    }

    for (pair_name in names(error_metrics_results)) {
        hub_results <- error_metrics_results[[pair_name]]
        
        hub_results[[names(hub_results)[1]]] <- add_star_to_min_post_calculation(hub_results[[names(hub_results)[1]]])
        hub_results[[names(hub_results)[2]]] <- add_star_to_min_post_calculation(hub_results[[names(hub_results)[2]]])

        error_metrics_results[[pair_name]] <- hub_results
    }

    print(error_metrics_results)


}

diebold_mariano <- function(hub_pairs, models, horizon, window_size) {

    dm_results <- list()

    for (pair in hub_pairs) {
    hub1_name <- pair[1]
    hub2_name <- pair[2]
    
    hub1_df <- data.frame(matrix(NA, nrow = length(models), ncol = length(models)))
    colnames(hub1_df) <- models
    rownames(hub1_df) <- models
    hub2_df <- hub1_df
    
    for (i in 1:length(models)) {
        for (j in 1:length(models)) {
        model1 <- models[i]
        model2 <- models[j]
        
        
        if (model1 == model2) {
            hub1_df[i, j] <- NA
            hub2_df[i, j] <- NA
        } else {

            if (grepl("arima", model1) || grepl("naive", model1)) {
                model1_hub1_filename <- paste0("../predictions/test/predictions/", hub1_name, "_h", horizon, "_w", window_size, "_", model1, "_predictions.csv")
                model1_hub2_filename <- paste0("../predictions/test/predictions/", hub2_name, "_h", horizon, "_w", window_size, "_", model1, "_predictions.csv")
                model1_hub1_predictions <- read.csv(model1_hub1_filename)
                model1_hub2_predictions <- read.csv(model1_hub2_filename)
                model1_predictions <- cbind(model1_hub1_predictions, model1_hub2_predictions)
            }
            else {
                model1_filename <- paste0("../predictions/test/predictions/", hub1_name, "_", hub2_name, "_h", horizon, "_w", window_size, "_", model1, "_predictions.csv")
                model1_predictions <- read.csv(model1_filename)
            }

            if (grepl("arima", model2) || grepl("naive", model2)) {
                model2_hub1_filename <- paste0("../predictions/test/predictions/", hub1_name, "_h", horizon, "_w", window_size, "_", model2, "_predictions.csv")
                model2_hub2_filename <- paste0("../predictions/test/predictions/", hub2_name, "_h", horizon, "_w", window_size, "_", model2, "_predictions.csv")
                model2_hub1_predictions <- read.csv(model2_hub1_filename)
                model2_hub2_predictions <- read.csv(model2_hub2_filename)
                model2_predictions <- cbind(model2_hub1_predictions, model2_hub2_predictions)
            }
            else {
                model2_filename <- paste0("../predictions/test/predictions/", hub1_name, "_", hub2_name, "_h", horizon, "_w", window_size, "_", model2, "_predictions.csv")
                model2_predictions <- read.csv(model2_filename)
            }

            actuals_hub1_filename <- paste0("../data/interpolated/", hub1_name, "_close_interpolated.csv")
            actuals_hub1 <- tail(read.csv(actuals_hub1_filename), horizon)
            colnames(actuals_hub1) <- c("Date", hub1_name)
            
            actuals_hub2_filename <- paste0("../data/interpolated/", hub2_name, "_close_interpolated.csv")
            actuals_hub2 <- tail(read.csv(actuals_hub2_filename),horizon)
            colnames(actuals_hub2) <- c("Date", hub2_name)
            
            model1_hub1_resids <- actuals_hub1[[hub1_name]] - model1_predictions[[hub1_name]]
            model2_hub1_resids <- actuals_hub1[[hub1_name]] - model2_predictions[[hub1_name]]
            
            model1_hub2_resids <- actuals_hub2[[hub2_name]] - model1_predictions[[hub2_name]]
            model2_hub2_resids <- actuals_hub2[[hub2_name]] - model2_predictions[[hub2_name]]
            
            hub1_dm <- dm.test(model1_hub1_resids, model2_hub1_resids, h = window_size, power = 1, alternative = "greater")
            hub1_df[j, i] <- hub1_dm$p.value
        

            hub2_dm <- dm.test(model1_hub2_resids, model2_hub2_resids, h = window_size, power = 1, alternative = "greater")
            hub2_df[j, i] <- hub2_dm$p.value
            
            
        }
        }
    }
    dm_hub_results <- list(hub1_df = hub1_df, hub2_df = hub2_df)
    names(dm_hub_results) <- c(hub1_name, hub2_name)
    dm_results[[paste(hub1_name, hub2_name, sep = "_")]] <- dm_hub_results
    }


    format_df_with_stars <- function(df, digits = 3) {
    df[] <- lapply(df, function(x) {
        sapply(x, function(value) {
        formatted_value <- formatC(value, format = "f", digits = digits)
        if (!is.na(value)) {
            
            if (value <= 0.001) {
            return(paste0(formatted_value, "***"))
            } else if (value <= 0.01) {
            return(paste0(formatted_value, "** "))
            } else if (value <= 0.051) {
            return(paste0(formatted_value, "*  "))
            } else {
            return(paste0(formatted_value, "   "))
            }
        } else {
            return("-   ")
        }
        })
    })
    return(df)
    }

    dm_results_formatted_with_stars <- lapply(dm_results, function(pair) {
    lapply(pair, format_df_with_stars, digits = 3)
    })

    print(dm_results_formatted_with_stars)

}

diebold_mariano_spread <- function(hub_pairs, models, horizon, window_size) {

    dm_results <- list()

    for (pair in hub_pairs) {
    hub1_name <- pair[1]
    hub2_name <- pair[2]
    
    hub1_df <- data.frame(matrix(NA, nrow = length(models), ncol = length(models)))
    colnames(hub1_df) <- models
    rownames(hub1_df) <- models
    hub2_df <- hub1_df
    
    for (i in 1:length(models)) {
        for (j in 1:length(models)) {
        model1 <- models[i]
        model2 <- models[j]
        
        
        if (model1 == model2) {
            hub1_df[i, j] <- NA
            hub2_df[i, j] <- NA
        } else {

            if (grepl("arima", model1) || grepl("naive", model1)) {
                model1_hub1_filename <- paste0("../predictions/test/predictions/", hub1_name, "_h", horizon, "_w", window_size, "_", model1, "_predictions.csv")
                model1_hub2_filename <- paste0("../predictions/test/predictions/", hub2_name, "_h", horizon, "_w", window_size, "_", model1, "_predictions.csv")
                model1_hub1_predictions <- read.csv(model1_hub1_filename)
                model1_hub2_predictions <- read.csv(model1_hub2_filename)
                model1_predictions <- cbind(model1_hub1_predictions, model1_hub2_predictions)
            }
            else {
                model1_filename <- paste0("../predictions/test/predictions/", hub1_name, "_", hub2_name, "_h", horizon, "_w", window_size, "_", model1, "_predictions.csv")
                model1_predictions <- read.csv(model1_filename)
            }

            if (grepl("arima", model2) || grepl("naive", model2)) {
                model2_hub1_filename <- paste0("../predictions/test/predictions/", hub1_name, "_h", horizon, "_w", window_size, "_", model2, "_predictions.csv")
                model2_hub2_filename <- paste0("../predictions/test/predictions/", hub2_name, "_h", horizon, "_w", window_size, "_", model2, "_predictions.csv")
                model2_hub1_predictions <- read.csv(model2_hub1_filename)
                model2_hub2_predictions <- read.csv(model2_hub2_filename)
                model2_predictions <- cbind(model2_hub1_predictions, model2_hub2_predictions)
            }
            else {
                model2_filename <- paste0("../predictions/test/predictions/", hub1_name, "_", hub2_name, "_h", horizon, "_w", window_size, "_", model2, "_predictions.csv")
                model2_predictions <- read.csv(model2_filename)
            }

            actuals_hub1_filename <- paste0("../data/interpolated/", hub1_name, "_close_interpolated.csv")
            hub1_historical_data <- read.csv(actuals_hub1_filename)
            colnames(hub1_historical_data) <- c("Date", hub1_name)
            actuals_hub1 <- tail(hub1_historical_data, horizon)
            
                
            actuals_hub2_filename <- paste0("../data/interpolated/", hub2_name, "_close_interpolated.csv")
            hub2_historical_data <- read.csv(actuals_hub2_filename)
            colnames(hub2_historical_data) <- c("Date", hub2_name)
            actuals_hub2 <- tail(hub2_historical_data, horizon)

            

            hub1_actual_returns <- log(actuals_hub1[[hub1_name]] / tail(lag(hub1_historical_data[[hub1_name]], window_size), horizon))
            hub2_actual_returns <- log(actuals_hub2[[hub2_name]] / tail(lag(hub2_historical_data[[hub2_name]], window_size), horizon))

            model1_hub1_predicted_returns <- log(model1_predictions[[hub1_name]] / tail(lag(hub1_historical_data[[hub1_name]], window_size), horizon))
            model1_hub2_predicted_returns <- log(model1_predictions[[hub2_name]] / tail(lag(hub2_historical_data[[hub2_name]], window_size), horizon))

            model2_hub1_predicted_returns <- log(model2_predictions[[hub1_name]] / tail(lag(hub1_historical_data[[hub1_name]], window_size), horizon))
            model2_hub2_predicted_returns <- log(model2_predictions[[hub2_name]] / tail(lag(hub2_historical_data[[hub2_name]], window_size), horizon))

            actual_spread <- hub1_actual_returns - hub2_actual_returns
            model1_predicted_spread <- model1_hub1_predicted_returns - model1_hub2_predicted_returns
            model2_predicted_spread <- model2_hub1_predicted_returns - model2_hub2_predicted_returns
                
            model1_spread_resids <- actual_spread - model1_predicted_spread
            model2_spread_resids <- actual_spread - model2_predicted_spread

            
            hub1_dm <- dm.test(model1_spread_resids, model2_spread_resids, h = window_size, power = 1, alternative = "greater")
            hub1_df[j, i] <- hub1_dm$p.value
        

            hub2_dm <- dm.test(model1_spread_resids, model2_spread_resids, h = window_size, power = 1, alternative = "greater")
            hub2_df[j, i] <- hub2_dm$p.value
            
            
        }
        }
    }
    dm_hub_results <- list(hub1_df = hub1_df, hub2_df = hub2_df)
    names(dm_hub_results) <- c(hub1_name, hub2_name)
    dm_results[[paste(hub1_name, hub2_name, sep = "_")]] <- dm_hub_results
    }


    format_df_with_stars <- function(df, digits = 3) {
    df[] <- lapply(df, function(x) {
        sapply(x, function(value) {
        formatted_value <- formatC(value, format = "f", digits = digits)
        if (!is.na(value)) {
            
            if (value <= 0.001) {
            return(paste0(formatted_value, "***"))
            } else if (value <= 0.01) {
            return(paste0(formatted_value, "** "))
            } else if (value <= 0.051) {
            return(paste0(formatted_value, "*  "))
            } else {
            return(paste0(formatted_value, "   "))
            }
        } else {
            return("-   ")
        }
        })
    })
    return(df)
    }

    dm_results_formatted_with_stars <- lapply(dm_results, function(pair) {
    lapply(pair, format_df_with_stars, digits = 3)
    })

    print(dm_results_formatted_with_stars)

}