import pandas as pd
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_percentage_error


def ann_system(hub1_name, hub2_name, validation_size, test_size, window_size, params, verbose=True, save=True):
    keras.utils.set_random_seed(42)
    hub1 = pd.read_csv(f"../../data/interpolated/{hub1_name}_close_interpolated.csv")
    hub2 = pd.read_csv(f"../../data/interpolated/{hub2_name}_close_interpolated.csv")

    hub1 = hub1.rename(columns={"CLOSE": "hub1_CLOSE"})
    hub2 = hub2.rename(columns={"CLOSE": "hub2_CLOSE"})
    hub1_hub2_diff = pd.DataFrame(hub1["hub1_CLOSE"] - hub2["hub2_CLOSE"], columns=["hub1_hub2_diff"], index=hub1.index)

    # Shift columns and store in new columns for hub1, hub2, and hub1_hub2_diff
    for i in range(window_size, window_size + params['lags'] + 1):
        hub1[f"hub1_CLOSE-{i- window_size}"] = hub1["hub1_CLOSE"].shift(i)
        hub2[f"hub2_CLOSE-{i - window_size}"] = hub2["hub2_CLOSE"].shift(i)
        hub1_hub2_diff[f"hub1_hub2_diff-{i - window_size}"] = hub1_hub2_diff["hub1_hub2_diff"].shift(i)

    # Concatenate and drop NaN rows in one step
    data = pd.concat([hub1, hub2, hub1_hub2_diff], axis=1).dropna()

    features = [
        'hub1_CLOSE-0', #'hub1_CLOSE-1', #'hub1_CLOSE-2', 'hub1_CLOSE-3', 'hub1_CLOSE-4', 'hub1_CLOSE-5',
        'hub2_CLOSE-0', #'hub2_CLOSE-1', #'hub2_CLOSE-2', 'hub2_CLOSE-3', 'hub2_CLOSE-4', 'hub2_CLOSE-5',
        'hub1_hub2_diff-0', 'hub1_hub2_diff-1', 'hub1_hub2_diff-2', 'hub1_hub2_diff-3', 'hub1_hub2_diff-4', 'hub1_hub2_diff-5',# 'hub1_hub2_diff-6'
    ]

    X = data[features].values

    y = data[['hub1_CLOSE', 'hub2_CLOSE']].values

    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    X_train, X_val = X_train[:-validation_size], X_train[-validation_size:]
    y_train, y_val = y_train[:-validation_size], y_train[-validation_size:]

    batch_size = params['batch_size']
    lr = params['lr']
    units = params['units']

    # Build a simple ANN model
    model = Sequential([
        Dense(units, activation='relu'),
        #Dense(1, activation='linear'),
        Dense(2)  # 2 outputs for both hubs
    ])
    #lr = 0.0038336791635842195 #TTF-THE
    #lr = 0.00026670003536052503 #TTF-NBP
    #lr = 0.0018046087209813773 #THE-NBP 2
    #lr = 0.003319584493448066 #THE-NBP
    #lr = 0.001
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=lr), loss='mape')

    # Train the model
    history = model.fit(X_train[:-window_size + 1], y_train[:-window_size + 1], epochs=25, batch_size=batch_size, 
                        #validation_data=(X_val, y_val), 
                        shuffle=False,
                        verbose=1)

    val_predictions = model.predict(X_val)
    

    val_dates = hub1[['Date']].values[-(validation_size+test_size):-test_size]
    val_predictions_df = pd.DataFrame(val_predictions, columns=[hub1_name, hub2_name])
    val_predictions_df['Date'] = val_dates.flatten()


    history = model.fit(np.vstack([X_train, X_val])[:-window_size + 1], np.vstack([y_train, y_val])[:-window_size + 1], shuffle = False, epochs=25, batch_size=batch_size, verbose=0)

    # Predict on test data
    test_predictions = model.predict(X_test)
    test_dates = hub1[['Date']].values[-test_size:]
    test_predictions_df = pd.DataFrame(test_predictions, columns=[hub1_name, hub2_name])
    test_predictions_df['Date'] = test_dates.flatten()



    if save:
        # Save the model
        val_predictions_df.to_csv(f"../../predictions/validation/predictions/{hub1_name}_{hub2_name}_v{validation_size}_h{test_size}_w{window_size}_ann_predictions.csv", index=False)
        test_predictions_df.to_csv(f"../../predictions/test/predictions/{hub1_name}_{hub2_name}_h{test_size}_w{window_size}_ann_predictions.csv", index=False)

        val_features_df = pd.DataFrame(X_val[:, :2], columns=[hub1_name, hub2_name])
        val_features_df['Date'] = val_dates.flatten()
        val_features_df.to_csv(f"../../predictions/validation/last_available/{hub1_name}_{hub2_name}_v{validation_size}_h{test_size}_w{window_size}_ann_last_available.csv", index=False)

        test_features_df = pd.DataFrame(X_test[:, :2], columns=[hub1_name, hub2_name])
        test_features_df['Date'] = test_dates.flatten()
        test_features_df.to_csv(f"../../predictions/test/last_available/{hub1_name}_{hub2_name}_h{test_size}_w{window_size}_ann_last_available.csv", index=False)

        val_actual_df = pd.DataFrame(y_val, columns=[hub1_name, hub2_name])
        val_actual_df['Date'] = val_dates.flatten()
        val_actual_df.to_csv(f"../../predictions/validation/actuals/{hub1_name}_{hub2_name}_v{validation_size}_h{test_size}_w{window_size}_ann_actuals.csv", index=False)

        test_actual_df = pd.DataFrame(y_test, columns=[hub1_name, hub2_name])
        test_actual_df['Date'] = test_dates.flatten()
        test_actual_df.to_csv(f"../../predictions/test/actuals/{hub1_name}_{hub2_name}_h{test_size}_w{window_size}_ann_actuals.csv", index=False)
    
    if verbose:
        # Calculate MAPE
        mape_hub1 = mean_absolute_percentage_error(y_test[:, 0], test_predictions[:, 0]) * 100
        print(f"MAPE for {hub1_name}: {mape_hub1:.2f}%")

        # Calculate MAPE for Hub 2
        mape_hub2 = mean_absolute_percentage_error(y_test[:, 1], test_predictions[:, 1]) * 100
        print(f"MAPE for {hub2_name}: {mape_hub2:.2f}%")



    return hub1, hub2