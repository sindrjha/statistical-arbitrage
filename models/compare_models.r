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

compare_error_metrics <- function(hub_pairs, models, horizon, window_size) {


    error_metrics <- c("mape", "rmse")


    # Initialize an empty list to store the results
    error_metrics_results <- list()

    # Iterate over hub pairs
    for (pair in hub_pairs) {
    hub1_name <- pair[1]
    hub2_name <- pair[2]
    
    # Initialize empty data frames to store the error metrics for each hub in the pair
    hub1_df <- data.frame(matrix(NA, nrow = length(models), ncol = length(error_metrics)))
    colnames(hub1_df) <- error_metrics
    rownames(hub1_df) <- models
    
    hub2_df <- hub1_df
    
    # Iterate over models for comparison
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
        
        # Construct file names for predictions and actuals
        actuals_hub1_filename <- paste0("../data/interpolated/", hub1_name, "_close_interpolated.csv")
        actuals_hub1 <- tail(read.csv(actuals_hub1_filename), horizon)
        colnames(actuals_hub1) <- c("Date", hub1_name)
            
        actuals_hub2_filename <- paste0("../data/interpolated/", hub2_name, "_close_interpolated.csv")
        actuals_hub2 <- tail(read.csv(actuals_hub2_filename),horizon)
        colnames(actuals_hub2) <- c("Date", hub2_name)


        
        # Calculate MAE and RMSE for hub1
        model_mape_hub1 <- mape(actuals_hub1[[hub1_name]], model1_predictions[[hub1_name]])*100
        model_rmse_hub1 <- rmse(actuals_hub1[[hub1_name]], model1_predictions[[hub1_name]])

        # Calculate MAE and RMSE for hub2
        model_mape_hub2 <- mape(actuals_hub2[[hub2_name]], model1_predictions[[hub2_name]])*100
        model_rmse_hub2 <- rmse(actuals_hub2[[hub2_name]], model1_predictions[[hub2_name]])
        
        # Store the metrics for hub1
        hub1_df[i, "mape"] <- round(model_mape_hub1, 2)
        hub1_df[i, "rmse"] <- round(model_rmse_hub1, 2)
        
        # Store the metrics for hub2
        hub2_df[i, "mape"] <- round(model_mape_hub2, 2)
        hub2_df[i, "rmse"] <- round(model_rmse_hub2, 2)
    }

    hub_results <- list(hub1 = hub1_df, hub2 = hub2_df)
    names(hub_results) <- c(hub1_name, hub2_name)
    # Store the results for this pair, using actual hub names
    error_metrics_results[[paste(hub1_name, hub2_name, sep = "_")]] <- hub_results
    }

    add_star_to_min_post_calculation <- function(df) {
    for (metric in colnames(df)) {
        min_value <- min(as.numeric(df[[metric]]), na.rm = TRUE) # Find the minimum value for each metric
        df[[metric]] <- sapply(df[[metric]], function(x) {
        numeric_x <- as.numeric(x) # Convert formatted strings back to numeric
        if (!is.na(numeric_x) && numeric_x == min_value) {
            paste0(format(numeric_x, digits = 3), "~")
        } else {
            paste0(format(numeric_x, digits = 3), " ")
        }
        })
    }
    return(df)
    }

    # Iterate over hub pairs in error_metrics_results and apply the star-adding function
    for (pair_name in names(error_metrics_results)) {
        hub_results <- error_metrics_results[[pair_name]]
        
        # Add stars to both hub1 and hub2
        hub_results[[names(hub_results)[1]]] <- add_star_to_min_post_calculation(hub_results[[names(hub_results)[1]]])
        hub_results[[names(hub_results)[2]]] <- add_star_to_min_post_calculation(hub_results[[names(hub_results)[2]]])
        
        # Update the results in the main list
        error_metrics_results[[pair_name]] <- hub_results
    }

    # Example: Print the results for one of the hub pairs (e.g., "the_nbp")
    print(error_metrics_results)


}

diebold_mariano <- function(hub_pairs, models, horizon, window_size) {

    # Initialize an empty list to store the results
    dm_results <- list()

    # Iterate over hub pairs
    for (pair in hub_pairs) {
    hub1_name <- pair[1]
    hub2_name <- pair[2]
    
    # Initialize empty data frames to store the p-values for each hub in the pair
    hub1_df <- data.frame(matrix(NA, nrow = length(models), ncol = length(models)))
    colnames(hub1_df) <- models
    rownames(hub1_df) <- models
    hub2_df <- hub1_df
    
    # Iterate over models for comparison
    for (i in 1:length(models)) {
        for (j in 1:length(models)) {
        model1 <- models[i]
        model2 <- models[j]
        
        
        # If models are the same, set value to NA or '-'
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
            
            # Calculate residuals for hub1
            model1_hub1_resids <- actuals_hub1[[hub1_name]] - model1_predictions[[hub1_name]]
            model2_hub1_resids <- actuals_hub1[[hub1_name]] - model2_predictions[[hub1_name]]
            
            # Calculate residuals for hub2
            model1_hub2_resids <- actuals_hub2[[hub2_name]] - model1_predictions[[hub2_name]]
            model2_hub2_resids <- actuals_hub2[[hub2_name]] - model2_predictions[[hub2_name]]
            
            # Perform Diebold-Mariano test for hub1 and store p-value
            hub1_dm <- dm.test(model1_hub1_resids, model2_hub1_resids, h = window_size, power = 1, alternative = "greater")
            hub1_df[j, i] <- hub1_dm$p.value
        

            # Perform Diebold-Mariano test for hub2 and store p-value
            hub2_dm <- dm.test(model1_hub2_resids, model2_hub2_resids, h = window_size, power = 1, alternative = "greater")
            hub2_df[j, i] <- hub2_dm$p.value
            
            
        }
        }
    }
    dm_hub_results <- list(hub1_df = hub1_df, hub2_df = hub2_df)
    names(dm_hub_results) <- c(hub1_name, hub2_name)
    # Store the results for this pair, using actual hub names
    dm_results[[paste(hub1_name, hub2_name, sep = "_")]] <- dm_hub_results
    }


    # Define a helper function to format numeric values and add stars based on p-value thresholds
    format_df_with_stars <- function(df, digits = 3) {
    df[] <- lapply(df, function(x) {
        # Apply the formatting and star addition to each element
        sapply(x, function(value) {
        formatted_value <- formatC(value, format = "f", digits = digits)
        if (!is.na(value)) {
            
            # Add stars based on the value
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

    # Apply the formatting and star-adding function to the results
    dm_results_formatted_with_stars <- lapply(dm_results, function(pair) {
    lapply(pair, format_df_with_stars, digits = 3)
    })

    # Print the formatted results with stars for "the_nbp"
    print(dm_results_formatted_with_stars)

}