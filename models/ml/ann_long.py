import pandas as pd
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_percentage_error

def ann_system_rolling(hub1_name, hub2_name, validation_size, test_size, window_size, params, verbose=True, save=True):
    
    hub1 = pd.read_csv(f"../../data/interpolated/{hub1_name}_close_interpolated.csv")
    hub2 = pd.read_csv(f"../../data/interpolated/{hub2_name}_close_interpolated.csv")

    hub1 = hub1.rename(columns={"CLOSE": "hub1_CLOSE"})
    hub2 = hub2.rename(columns={"CLOSE": "hub2_CLOSE"})
    hub1_hub2_diff = pd.DataFrame(hub1["hub1_CLOSE"] - hub2["hub2_CLOSE"], columns=["hub1_hub2_diff"], index=hub1.index)

    # Shift columns and store in new columns for hub1, hub2, and hub1_hub2_diff
    for i in range(window_size, window_size + params['lags'] + 1):
        hub1[f"hub1_CLOSE-{i - window_size}"] = hub1["hub1_CLOSE"].shift(i)
        hub2[f"hub2_CLOSE-{i - window_size}"] = hub2["hub2_CLOSE"].shift(i)
        hub1_hub2_diff[f"hub1_hub2_diff-{i - window_size}"] = hub1_hub2_diff["hub1_hub2_diff"].shift(i)

    # Concatenate and drop NaN rows in one step
    data = pd.concat([hub1, hub2, hub1_hub2_diff], axis=1).dropna()

    features = [
        'hub1_CLOSE-0', 'hub2_CLOSE-0',
        'hub1_hub2_diff-0', 'hub1_hub2_diff-1', 'hub1_hub2_diff-2', 'hub1_hub2_diff-3', 'hub1_hub2_diff-4', 'hub1_hub2_diff-5'
    ]

    X = data[features].values
    y = data[['hub1_CLOSE', 'hub2_CLOSE']].values

    lr = params['lr']
    units = params['units']

    # Storage for predictions and actual values
    val_predictions = []
    test_predictions = []

    # Separate validation and test periods
    X_val, y_val = X[-(validation_size + test_size):-test_size], y[-(validation_size + test_size):-test_size]
    X_test, y_test = X[-test_size:], y[-test_size:]

    

    # Rolling prediction for validation data
    for i in range(len(X_val)):
        keras.utils.set_random_seed(42)
        model = Sequential([
            Dense(units, activation='relu'),
            Dense(2)
        ])
        model.compile(optimizer=Adam(learning_rate=lr), loss='mape')
        # Train on all data up to the current validation point, accounting for the window size offset
        X_train_current = X[:-(validation_size + test_size) + i - window_size + 1]
        y_train_current = y[:-(validation_size + test_size) + i - window_size + 1]

        model.fit(X_train_current, y_train_current, epochs=25, batch_size=1, verbose=0, shuffle=False)
        val_predictions.append(model.predict(X_val[i:i+1]))

    # Convert validation predictions to array for evaluation or saving
    val_predictions = np.array(val_predictions).squeeze()
    val_dates = hub1[['Date']].values[-(validation_size + test_size):-test_size]
    val_predictions_df = pd.DataFrame(val_predictions, columns=[hub1_name, hub2_name])
    val_predictions_df['Date'] = val_dates.flatten()

    # Rolling prediction for test data
    for i in range(len(X_test)):
        keras.utils.set_random_seed(42)
        model = Sequential([
            Dense(units, activation='relu'),
            Dense(2)
        ])
        model.compile(optimizer=Adam(learning_rate=lr), loss='mape')
        # Train on all data up to the current test point, with the window size offset
        X_train_current = X[:-test_size + i - window_size + 1]
        y_train_current = y[:-test_size + i - window_size + 1]

        model.fit(X_train_current, y_train_current, epochs=25, batch_size=1, verbose=0, shuffle=False)
        test_predictions.append(model.predict(X_test[i:i+1]))

    test_predictions = np.array(test_predictions).squeeze()
    test_dates = hub1[['Date']].values[-test_size:]
    test_predictions_df = pd.DataFrame(test_predictions, columns=[hub1_name, hub2_name])
    test_predictions_df['Date'] = test_dates.flatten()

    if save:
        val_predictions_df.to_csv(f"../../predictions/validation/predictions/{hub1_name}_{hub2_name}_v{validation_size}_h{test_size}_w{window_size}_ann_long_predictions.csv", index=False)
        test_predictions_df.to_csv(f"../../predictions/test/predictions/{hub1_name}_{hub2_name}_h{test_size}_w{window_size}_ann_long_predictions.csv", index=False)

        val_features_df = pd.DataFrame(X_val[:, :2], columns=[hub1_name, hub2_name])
        val_features_df['Date'] = val_dates.flatten()
        val_features_df.to_csv(f"../../predictions/validation/last_available/{hub1_name}_{hub2_name}_v{validation_size}_h{test_size}_w{window_size}_ann_long_last_available.csv", index=False)

        test_features_df = pd.DataFrame(X_test[:, :2], columns=[hub1_name, hub2_name])
        test_features_df['Date'] = test_dates.flatten()
        test_features_df.to_csv(f"../../predictions/test/last_available/{hub1_name}_{hub2_name}_h{test_size}_w{window_size}_ann_long_last_available.csv", index=False)

        val_actual_df = pd.DataFrame(y_val, columns=[hub1_name, hub2_name])
        val_actual_df['Date'] = val_dates.flatten()
        val_actual_df.to_csv(f"../../predictions/validation/actuals/{hub1_name}_{hub2_name}_v{validation_size}_h{test_size}_w{window_size}_ann_long_actuals.csv", index=False)

        test_actual_df = pd.DataFrame(y_test, columns=[hub1_name, hub2_name])
        test_actual_df['Date'] = test_dates.flatten()
        test_actual_df.to_csv(f"../../predictions/test/actuals/{hub1_name}_{hub2_name}_h{test_size}_w{window_size}_ann_long_actuals.csv", index=False)

    if verbose:
        # Calculate MAPE for validation and test
        mape_hub1_val = mean_absolute_percentage_error(y_val[:, 0], val_predictions[:, 0]) * 100
        mape_hub2_val = mean_absolute_percentage_error(y_val[:, 1], val_predictions[:, 1]) * 100
        mape_hub1_test = mean_absolute_percentage_error(y_test[:, 0], test_predictions[:, 0]) * 100
        mape_hub2_test = mean_absolute_percentage_error(y_test[:, 1], test_predictions[:, 1]) * 100
        print(f"Validation MAPE for {hub1_name}: {mape_hub1_val:.2f}%")
        print(f"Validation MAPE for {hub2_name}: {mape_hub2_val:.2f}%")
        print(f"Test MAPE for {hub1_name}: {mape_hub1_test:.2f}%")
        print(f"Test MAPE for {hub2_name}: {mape_hub2_test:.2f}%")

    return hub1, hub2
