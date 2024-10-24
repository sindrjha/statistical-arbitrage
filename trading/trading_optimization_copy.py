import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go


class OptimizeTradingSystem:
    def __init__(self, hub1_name, hub2_name, model, validation_size, test_size, window_size):
        self.hub1_name = hub1_name
        self.hub2_name = hub2_name
        self.model = model
        self.validation_size = validation_size
        self.test_size = test_size
        self.window_size = window_size
        
        self.hub1_predictions = None
        self.hub2_predictions = None
        self.hub1_last_available_data = None
        self.hub1_historical_data = None
        

        self.hub2_last_available_data = None
        self.hub1_actuals = None
        self.hub2_actuals = None
        self.hub2_historical_data = None

        self.returns_difference = None
        
        self.trades = None
        self.pl = None
        self.returns = None
        self.returns_with_transaction_costs = None
        self.profit = 0
        
        self._load_data()

    def _load_data(self):
        # Loading predictions and historical data based on the model type
        if "arima" in self.model:
            hub1_data = f"{self.hub1_name}_v{self.validation_size}_h{self.test_size}_w{self.window_size}"
            predictions = pd.read_csv(f"../predictions/validation/predictions/{hub1_data}_{self.model}_predictions.csv")
            predictions = predictions.set_index("Date")
            self.hub1_predictions = predictions[[self.hub1_name]]

            last_available = pd.read_csv(f"../predictions/validation/last_available/{hub1_data}_{self.model}_last_available.csv")
            last_available = last_available.set_index("Date")
            self.hub1_last_available_data = last_available[[self.hub1_name]]

            actuals = pd.read_csv(f"../predictions/validation/actuals/{hub1_data}_{self.model}_actuals.csv")
            actuals = actuals.set_index("Date")
            self.hub1_actuals = actuals[[self.hub1_name]]


            hub2_data = f"{self.hub2_name}_v{self.validation_size}_h{self.test_size}_w{self.window_size}"
            predictions = pd.read_csv(f"../predictions/validation/predictions/{hub2_data}_{self.model}_predictions.csv")
            predictions = predictions.set_index("Date")
            self.hub2_predictions = predictions[[self.hub2_name]]

            last_available = pd.read_csv(f"../predictions/validation/last_available/{hub2_data}_{self.model}_last_available.csv")
            last_available = last_available.set_index("Date")
            self.hub2_last_available_data = last_available[[self.hub2_name]]

            actuals = pd.read_csv(f"../predictions/validation/actuals/{hub2_data}_{self.model}_actuals.csv")
            actuals = actuals.set_index("Date")
            self.hub2_actuals = actuals[[self.hub2_name]]

        else:
            data = f"{self.hub1_name}_{self.hub2_name}_v{self.validation_size}_h{self.test_size}_w{self.window_size}"
            predictions = pd.read_csv(f"../predictions/validation/predictions/{data}_{self.model}_predictions.csv")
            predictions = predictions.set_index("Date")
            self.hub1_predictions = predictions[[self.hub1_name]]
            self.hub2_predictions = predictions[[self.hub2_name]]

            last_available = pd.read_csv(f"../predictions/validation/last_available/{data}_{self.model}_last_available.csv")
            last_available = last_available.set_index("Date")
            self.hub1_last_available_data = last_available[[self.hub1_name]]
            self.hub2_last_available_data = last_available[[self.hub2_name]]

            actuals = pd.read_csv(f"../predictions/validation/actuals/{data}_{self.model}_actuals.csv")
            actuals = actuals.set_index("Date")
            self.hub1_actuals = actuals[[self.hub1_name]]
            self.hub2_actuals = actuals[[self.hub2_name]]

        # Loading historical data
        hub1_historical_data = pd.read_csv(f"../data/interpolated/{self.hub1_name}_close_interpolated.csv")
        hub1_historical_data = hub1_historical_data.set_index("Date")
        hub1_historical_data.rename(columns={"CLOSE": self.hub1_name}, inplace=True)
        self.hub1_historical_data = hub1_historical_data
    
        hub2_historical_data = pd.read_csv(f"../data/interpolated/{self.hub2_name}_close_interpolated.csv")
        hub2_historical_data = hub2_historical_data.set_index("Date")
        hub2_historical_data.rename(columns={"CLOSE": self.hub2_name}, inplace=True)
        self.hub2_historical_data = hub2_historical_data


        # Calculating predicted returns
        self.hub1_predicted_returns = np.log(self.hub1_predictions.values / self.hub1_last_available_data.values) * 100
        self.hub2_predicted_returns = np.log(self.hub2_predictions.values / self.hub2_last_available_data.values) * 100

        # Historical returns difference
        hub1_historical_returns = np.log(hub1_historical_data[self.window_size:][[self.hub1_name]].values / hub1_historical_data[:-self.window_size][[self.hub1_name]].values) * 100
        hub2_historical_returns = np.log(hub2_historical_data[self.window_size:][[self.hub2_name]].values / hub2_historical_data[:-self.window_size][[self.hub2_name]].values) * 100
        self.returns_difference = hub1_historical_returns - hub2_historical_returns

    def run_trading_system(self, rolling_window=5, lower_threshold=0, upper_threshold = 3, naive=False, verbose=False, plot=False):
        np.random.seed(42)
        transaction_costs = 0.005
        self.trades = [0] * self.validation_size
        self.pl = [0] * self.validation_size    
        self.returns = [0] * self.validation_size
        self.returns_with_transaction_costs = [0] * self.validation_size
        self.profit = 0

        start = self.test_size + self.validation_size + self.window_size

        for i in range(self.validation_size):
            hub1_predicted_return = self.hub1_predicted_returns[i].item()
            hub2_predicted_return = self.hub2_predicted_returns[i].item()

            returns_difference = hub1_predicted_return - hub2_predicted_return
            rolling_data = self.returns_difference[-start - rolling_window + i: - start + i]
            returns_difference_std = rolling_data.std().item()

            r = np.random.rand() 

            if ((returns_difference > lower_threshold * returns_difference_std) and (returns_difference < upper_threshold * returns_difference_std))or (naive and r < 0.5):
                long_position_return = np.log(self.hub1_historical_data.values[-start + self.window_size + i].item() / self.hub1_historical_data.values[-start + i].item())
                short_position_return = -np.log(self.hub2_historical_data.values[-start + self.window_size + i].item() / self.hub2_historical_data.values[-start + i].item())

                # Transaction costs adjusted returns
                long_position_return_with_transaction_costs = np.log((self.hub1_historical_data.values[-start + self.window_size + i].item() - 2 * transaction_costs) / self.hub1_historical_data.values[-start + i].item())
                short_position_return_with_transaction_costs = -np.log((self.hub2_historical_data.values[-start + self.window_size + i].item() - 2 * transaction_costs) / self.hub2_historical_data.values[-start + i].item())

                # Net profits using historical data
                net_profit_long = self.hub1_historical_data.values[-start + i + self.window_size].item() - self.hub1_historical_data.values[-start + i].item()
                net_profit_short = self.hub2_historical_data.values[-start + i].item() - self.hub2_historical_data.values[-start + i + self.window_size].item()

                monetary_adjustment = (self.hub1_historical_data.values[-start + i].item() / self.hub2_historical_data.values[-start + i].item())
                # Update profit and store trades and returns
                self.profit += net_profit_long + monetary_adjustment * net_profit_short - transaction_costs * 4
                self.trades[i] = 1
                self.returns[i] = long_position_return + short_position_return
                self.returns_with_transaction_costs[i] = long_position_return_with_transaction_costs + short_position_return_with_transaction_costs

            elif ((returns_difference < -lower_threshold * returns_difference_std) and (returns_difference > -upper_threshold * returns_difference_std)) or (naive and r >= 0.5):
                long_position_return = np.log(self.hub2_historical_data.values[-start + self.window_size + i].item() / self.hub2_historical_data.values[-start + i].item())
                short_position_return = -np.log(self.hub1_historical_data.values[-start + self.window_size + i].item() / self.hub1_historical_data.values[-start + i].item())

                long_position_return_with_transaction_costs = np.log((self.hub2_historical_data.values[-start + self.window_size + i].item() - 2 * transaction_costs) / self.hub2_historical_data.values[-start + i].item())
                short_position_return_with_transaction_costs = -np.log((self.hub1_historical_data.values[-start + self.window_size + i].item() - 2 * transaction_costs) / self.hub1_historical_data.values[-start + i].item())

                net_profit_long = self.hub2_historical_data.values[-start + i + self.window_size].item() - self.hub2_historical_data.values[-start + i].item()
                net_profit_short = self.hub1_historical_data.values[-start + i].item() - self.hub1_historical_data.values[-start + i + self.window_size].item()

                monetary_adjustment = (self.hub2_historical_data.values[-start + i].item() / self.hub1_historical_data.values[-start + i].item())

                self.profit += net_profit_long +  monetary_adjustment*net_profit_short - transaction_costs * 4
                self.trades[i] = 1
                self.returns[i] = long_position_return + short_position_return
                self.returns_with_transaction_costs[i] = long_position_return_with_transaction_costs + short_position_return_with_transaction_costs


            self.pl[i] = self.profit

        if verbose:

            mean_returns, std_returns, mean_returns_with_tc, ci_returns, ci_returns_with_tc = self.get_returns_stats()
            print(f"Profit: {self.profit:.2f}")
            print()
            print(f"Mean returns: {mean_returns:.2f}%")
            print(f"Standard deviation of returns: {std_returns:.2f}%")
            print(f"Sharpe ratio: {mean_returns/std_returns:.2f}")
            print(f"Mean returns with transaction costs: {mean_returns_with_tc:.2f}%")
            print(f"Confidence interval (returns): {ci_returns[0]:.2f}% - {ci_returns[1]:.2f}%")
            print(f"Confidence interval (returns with transaction costs): {ci_returns_with_tc[0]:.2f}% - {ci_returns_with_tc[1]:.2f}%")
            print()
            trade_rates = self.get_trade_rates()
            print(f"Win rate for returns: {trade_rates['win_rate_returns']:.2%}")
            print(f"No trade rate for returns: {trade_rates['no_trade_rate_returns']:.2%}")
            print(f"Loss rate for returns: {trade_rates['loss_rate_returns']:.2%}")
            print()
            print(f"Win rate for returns with transaction costs: {trade_rates['win_rate_returns_with_tc']:.2%}")
            print(f"No trade rate for returns with transaction costs: {trade_rates['no_trade_rate_returns_with_tc']:.2%}")
            print(f"Loss rate for returns with transaction costs: {trade_rates['loss_rate_returns_with_tc']:.2%}")

        if plot:
            fig = go.Figure()

            # Add line plot for pl
            fig.add_trace(go.Scatter(
                x=self.hub1_actuals.index,
                y=self.pl,
                mode='lines',
                name='Profit/Loss'
            ))

            # Add scatter plot for trades
            fig.add_trace(go.Scatter(
                x=self.hub1_actuals.index,
                y=self.trades,
                mode='markers',
                name='Trades',
                yaxis='y2'
            ))

            # Create axis objects
            fig.update_layout(
                title='Profit/Loss and Trades Over Time',
                xaxis=dict(title='Time'),
                yaxis=dict(title='Profit/Loss'),
                yaxis2=dict(title='Trades', overlaying='y', side='right')
            )

            fig.show()

    def study(self, rolling_window_range=[5,60,5], lower_threshold_range=[0,2,0.2], upper_threshold_range=[3,4,0.5], criteria = "profit", min_trades=10, verbose=False, plot=False):
        best_profit = 0
        best_lower_threshold = 0
        best_upper_threshold = 0
        best_rolling_window = 0

        rolling_window_l, rolling_window_u, rolling_window_s = rolling_window_range
        l_threshold_l, l_threshold_u, l_threshold_s = lower_threshold_range
        u_threshold_l, u_threshold_u, u_threshold_s = upper_threshold_range

        for lower_threshold in np.arange(l_threshold_l, l_threshold_u, l_threshold_s):
            for upper_threshold in np.arange(u_threshold_l, u_threshold_u, u_threshold_s):
                for rolling_window in np.arange(rolling_window_l, rolling_window_u, rolling_window_s):
                        self.run_trading_system(rolling_window=rolling_window, lower_threshold=lower_threshold, upper_threshold=upper_threshold)
                        mean_returns, std_returns, _, _, _ = self.get_returns_stats()
                        if criteria == "profit":
                            if (self.profit > best_profit) and (self.get_num_trades() > min_trades):
                                best_profit = self.profit
                                best_lower_threshold = lower_threshold
                                best_upper_threshold = upper_threshold
                                best_rolling_window = rolling_window
                        elif criteria == "mean_returns":
                            if (mean_returns > best_profit) and (self.get_num_trades() > min_trades):
                                best_profit = mean_returns
                                best_lower_threshold = lower_threshold
                                best_upper_threshold = upper_threshold
                                best_rolling_window = rolling_window
                        elif criteria == "sharpe":
                            sharpe = mean_returns / std_returns if std_returns != 0 else 0
                            if (sharpe > best_profit) and (self.get_num_trades() > min_trades):
                                best_profit = sharpe
                                best_lower_threshold = lower_threshold
                                best_upper_threshold = upper_threshold
                                best_rolling_window = rolling_window

       

        if verbose or plot:
            print(f"Best rolling window: {best_rolling_window:.2f}, Best lower threshold: {best_lower_threshold:.2f}, Best upper threshold: {best_upper_threshold:.2f}")
            self.run_trading_system(rolling_window=best_rolling_window, lower_threshold=best_lower_threshold, upper_threshold=best_upper_threshold, verbose=verbose, plot=plot)

        return best_profit, best_rolling_window, best_lower_threshold, best_upper_threshold 

    def get_profit(self):
        return self.profit
    
    def get_trades(self):
        return self.trades
    
    def get_num_trades(self):
        return sum(self.trades)
    
    def get_pl(self):
        return self.pl

    def get_returns_stats(self):
        mean_returns = np.mean(self.returns) * 100
        std_returns = np.std(self.returns) * 100
        mean_returns_with_tc = np.mean(self.returns_with_transaction_costs) * 100
        standard_error_returns = np.std(self.returns) / np.sqrt(len(self.returns))
        ci_returns = (
            (np.mean(self.returns) - 1.96 * standard_error_returns) * 100,
            (np.mean(self.returns) + 1.96 * standard_error_returns) * 100
        )
        standard_error_returns_with_tc = np.std(self.returns_with_transaction_costs) / np.sqrt(len(self.returns_with_transaction_costs))
        ci_returns_with_tc = (
            (np.mean(self.returns_with_transaction_costs) - 1.96 * standard_error_returns_with_tc) * 100,
            (np.mean(self.returns_with_transaction_costs) + 1.96 * standard_error_returns_with_tc) * 100
        )
        
        return mean_returns, std_returns, mean_returns_with_tc, ci_returns, ci_returns_with_tc

    def get_trade_rates(self):
        # Calculate win, no-trade, and loss rates for returns
        win_rate_returns = sum(1 for r in self.returns if r > 0) / len(self.returns)
        no_trade_rate_returns = sum(1 for r in self.returns if r == 0) / len(self.returns)
        loss_rate_returns = sum(1 for r in self.returns if r < 0) / len(self.returns)

        # Calculate win, no-trade, and loss rates for returns with transaction costs
        win_rate_returns_with_tc = sum(1 for r in self.returns_with_transaction_costs if r > 0) / len(self.returns_with_transaction_costs)
        no_trade_rate_returns_with_tc = sum(1 for r in self.returns_with_transaction_costs if r == 0) / len(self.returns_with_transaction_costs)
        loss_rate_returns_with_tc = sum(1 for r in self.returns_with_transaction_costs if r < 0) / len(self.returns_with_transaction_costs)

        return {
            'win_rate_returns': win_rate_returns,
            'no_trade_rate_returns': no_trade_rate_returns,
            'loss_rate_returns': loss_rate_returns,
            'win_rate_returns_with_tc': win_rate_returns_with_tc,
            'no_trade_rate_returns_with_tc': no_trade_rate_returns_with_tc,
            'loss_rate_returns_with_tc': loss_rate_returns_with_tc
        }

