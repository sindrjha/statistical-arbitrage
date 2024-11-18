import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go


class TradingSystem:
    def __init__(self, hub1_name, hub2_name, model, validation_size, test_size, window_size, mode = "validation", strategy = "forecasting", cointegration_rolling_window = None):
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

        self.strategy = strategy
        
        self.trades = None
        self.pl = None
        self.returns = None
        self.returns_with_transaction_costs = None
        self.profit = 0

        self.mode = mode

        self.load_historical_data()
        
        if self.strategy == "cointegration":
            self._load_cointegration_data(cointegration_rolling_window)
        else:
            self._load_forecast_data()

    def load_historical_data(self):
        hub1_historical_data = pd.read_csv(f"../data/interpolated/{self.hub1_name}_close_interpolated.csv")
        hub1_historical_data = hub1_historical_data.set_index("Date")
        hub1_historical_data.rename(columns={"CLOSE": self.hub1_name}, inplace=True)
        self.hub1_historical_data = hub1_historical_data
    
        hub2_historical_data = pd.read_csv(f"../data/interpolated/{self.hub2_name}_close_interpolated.csv")
        hub2_historical_data = hub2_historical_data.set_index("Date")
        hub2_historical_data.rename(columns={"CLOSE": self.hub2_name}, inplace=True)
        self.hub2_historical_data = hub2_historical_data

    def _load_forecast_data(self):
        # Loading predictions and historical data based on the model type
        if "arima" in self.model:
            if self.mode == "validation":
                hub1_data = f"{self.hub1_name}_v{self.validation_size}_h{self.test_size}_w{self.window_size}"
                hub2_data = f"{self.hub2_name}_v{self.validation_size}_h{self.test_size}_w{self.window_size}"
            else:
                hub1_data = f"{self.hub1_name}_h{self.test_size}_w{self.window_size}"
                hub2_data = f"{self.hub2_name}_h{self.test_size}_w{self.window_size}"

            predictions = pd.read_csv(f"../predictions/{self.mode}/predictions/{hub1_data}_{self.model}_predictions.csv")
            predictions = predictions.set_index("Date")
            self.hub1_predictions = predictions[[self.hub1_name]]

            last_available = pd.read_csv(f"../predictions/{self.mode}/last_available/{hub1_data}_{self.model}_last_available.csv")
            last_available = last_available.set_index("Date")
            self.hub1_last_available_data = last_available[[self.hub1_name]]

            actuals = pd.read_csv(f"../predictions/{self.mode}/actuals/{hub1_data}_{self.model}_actuals.csv")
            actuals = actuals.set_index("Date")
            self.hub1_actuals = actuals[[self.hub1_name]]

            predictions = pd.read_csv(f"../predictions/{self.mode}/predictions/{hub2_data}_{self.model}_predictions.csv")
            predictions = predictions.set_index("Date")
            self.hub2_predictions = predictions[[self.hub2_name]]

            last_available = pd.read_csv(f"../predictions/{self.mode}/last_available/{hub2_data}_{self.model}_last_available.csv")
            last_available = last_available.set_index("Date")
            self.hub2_last_available_data = last_available[[self.hub2_name]]

            actuals = pd.read_csv(f"../predictions/{self.mode}/actuals/{hub2_data}_{self.model}_actuals.csv")
            actuals = actuals.set_index("Date")
            self.hub2_actuals = actuals[[self.hub2_name]]

        else:
            if self.mode == "validation":
                data = f"{self.hub1_name}_{self.hub2_name}_v{self.validation_size}_h{self.test_size}_w{self.window_size}"
            else:
                data = f"{self.hub1_name}_{self.hub2_name}_h{self.test_size}_w{self.window_size}"

            predictions = pd.read_csv(f"../predictions/{self.mode}/predictions/{data}_{self.model}_predictions.csv")
            predictions = predictions.set_index("Date")
            self.hub1_predictions = predictions[[self.hub1_name]]
            self.hub2_predictions = predictions[[self.hub2_name]]

            last_available = pd.read_csv(f"../predictions/{self.mode}/last_available/{data}_{self.model}_last_available.csv")
            last_available = last_available.set_index("Date")
            self.hub1_last_available_data = last_available[[self.hub1_name]]
            self.hub2_last_available_data = last_available[[self.hub2_name]]

            actuals = pd.read_csv(f"../predictions/{self.mode}/actuals/{data}_{self.model}_actuals.csv")
            actuals = actuals.set_index("Date")
            self.hub1_actuals = actuals[[self.hub1_name]]
            self.hub2_actuals = actuals[[self.hub2_name]]

        # Calculating predicted returns
        self.hub1_predicted_returns = np.log(self.hub1_predictions.values / self.hub1_last_available_data.values)
        self.hub2_predicted_returns = np.log(self.hub2_predictions.values / self.hub2_last_available_data.values)

        # Historical returns difference
        hub1_historical_returns = np.log(self.hub1_historical_data[self.window_size:][[self.hub1_name]].values / self.hub1_historical_data[:-self.window_size][[self.hub1_name]].values)
        hub2_historical_returns = np.log(self.hub2_historical_data[self.window_size:][[self.hub2_name]].values / self.hub2_historical_data[:-self.window_size][[self.hub2_name]].values)
        self.returns_difference = hub1_historical_returns - hub2_historical_returns

    def _load_cointegration_data(self, cointegration_rolling_window):

        cointegration_data = pd.read_csv(f"../predictions/cointegration/{self.hub1_name}_{self.hub2_name}_r{cointegration_rolling_window}_v{self.validation_size}_h{self.test_size}_w{self.window_size}_{self.model}_cointegration.csv")

        self.price_difference = cointegration_data[["residuals"]]
        self.cointegration_beta = cointegration_data[["beta"]]


    def run_trading_system(self, volatility = "rolling", rolling_window=5, lower_threshold=0,  special_strategy = "no", verbose=False, plot=False):
        np.random.seed(42)

        size = self.validation_size if self.mode == "validation" else self.test_size
        transaction_costs = 0.005
        self.trades = [0] * size
        self.pl = [0] * size
        self.returns = [0] * size
        self.returns_with_transaction_costs = [0] * size
        self.profit = 0

        if self.mode == "validation":
            start = self.test_size + self.validation_size + self.window_size
        else:
            start = self.test_size + self.window_size

        if volatility != "rolling":
            if self.mode == "validation":
                data = f"{self.hub1_name}_{self.hub2_name}_v{self.validation_size}_h{self.test_size}_w{self.window_size}"
            else:
                data = f"{self.hub1_name}_{self.hub2_name}_h{self.test_size}_w{self.window_size}"
            if self.strategy == "forecasting":
                garch_predictions = pd.read_csv(f"../predictions/{self.mode}/predictions/{data}_{volatility}_predictions.csv")
            else:
                garch_predictions = pd.read_csv(f"../predictions/{self.mode}/predictions/{data}_{volatility}_cointegration_predictions.csv")


        for i in range(size):

            if self.strategy == "forecasting":
                difference_series = self.returns_difference
                hub1_predicted_return = self.hub1_predicted_returns[i].item()
                hub2_predicted_return = self.hub2_predicted_returns[i].item()
                current_difference = hub1_predicted_return - hub2_predicted_return
                monetary_adjustment = (self.hub1_historical_data.values[-start + i].item() / self.hub2_historical_data.values[-start + i].item())
            else:
                difference_series = self.price_difference
                current_difference = self.price_difference.values[-start + i].item()
                cointegration_beta = self.cointegration_beta.values[-start + i].item()

            if volatility == "rolling":
                rolling_data = difference_series[-start - rolling_window + i: - start + i]
                std_dev = rolling_data.std().item()
            else:
                std_dev = garch_predictions["Sigma"].values[i].item()

            r = np.random.rand() 

            
            hub1_return = np.log(self.hub1_historical_data.values[-start + self.window_size + i].item() / self.hub1_historical_data.values[-start + i].item())
            hub2_return = np.log(self.hub2_historical_data.values[-start + self.window_size + i].item() / self.hub2_historical_data.values[-start + i].item())

            # Transaction costs adjusted returns
            hub1_return_with_transaction_costs = np.log((self.hub1_historical_data.values[-start + self.window_size + i].item() - 2 * transaction_costs) / self.hub1_historical_data.values[-start + i].item())
            hub2_return_with_transaction_costs = np.log((self.hub2_historical_data.values[-start + self.window_size + i].item() - 2 * transaction_costs) / self.hub2_historical_data.values[-start + i].item())

            # Net profits using historical data
            net_profit_hub1 = self.hub1_historical_data.values[-start + i + self.window_size].item() - self.hub1_historical_data.values[-start + i].item()
            net_profit_hub2 = self.hub2_historical_data.values[-start + i + self.window_size].item() - self.hub2_historical_data.values[-start + i].item()

            if (current_difference >  lower_threshold * std_dev) or (special_strategy == "naive" and difference_series[-start +i] > 0) or (special_strategy == "perfect_information" and difference_series[-start + self.window_size +i] > 0):
                
                if self.strategy == "forecasting":
                    # Go long on hub1 and short on hub2
                    self.profit += net_profit_hub1 - monetary_adjustment*net_profit_hub2 - transaction_costs * 4
                    self.returns[i] = hub1_return - hub2_return
                    self.returns_with_transaction_costs[i] = hub1_return_with_transaction_costs - hub2_return_with_transaction_costs
                else:
                    # Go short on hub1 and long on hub2
                    self.profit += cointegration_beta*net_profit_hub2 - net_profit_hub1 - transaction_costs * 4
                    self.returns[i] = cointegration_beta*hub2_return - hub1_return
                    self.returns_with_transaction_costs[i] = cointegration_beta*hub2_return_with_transaction_costs - hub1_return_with_transaction_costs

                self.trades[i] = 1

            elif (current_difference <  - lower_threshold * std_dev) or (special_strategy == "naive" and difference_series[-start +i] < 0) or (special_strategy == "perfect_information" and difference_series[-start + self.window_size +i] < 0):

                if self.strategy == "forecasting":
                    # Go short on hub1 and long on hub2
                    self.profit += monetary_adjustment*net_profit_hub2 - net_profit_hub1 - transaction_costs * 4
                    self.returns[i] = hub2_return - hub1_return
                    self.returns_with_transaction_costs[i] = hub2_return_with_transaction_costs - hub1_return_with_transaction_costs
                else:
                    # Go long on hub1 and short on hub2
                    self.profit += net_profit_hub1 - cointegration_beta*net_profit_hub2- transaction_costs * 4
                    self.returns[i] = hub1_return - cointegration_beta*hub2_return
                    self.returns_with_transaction_costs[i] = hub1_return_with_transaction_costs - cointegration_beta*hub2_return_with_transaction_costs

                self.trades[i] = 1
            


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
                x=self.hub1_historical_data.tail(self.test_size).index,
                y=self.pl,
                mode='lines',
                name='Profit/Loss'
            ))

            # Add scatter plot for trades
            fig.add_trace(go.Scatter(
                x=self.hub1_historical_data.tail(self.test_size).index,
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

    def study(self, volatility = "rolling", rolling_window_range=[5,60,5], lower_threshold_range=[0,2,0.2], criteria = "profit", min_trades=10, verbose=False, plot=False):

        
        l_threshold_l, l_threshold_u, l_threshold_s = lower_threshold_range

        if volatility == "rolling":
            rolling_window_l, rolling_window_u, rolling_window_s = rolling_window_range


        best_profit = -float('inf')
        best_lower_threshold = None
        best_upper_threshold = None
        best_rolling_window = None

        for lower_threshold in np.arange(l_threshold_l, l_threshold_u, l_threshold_s):
                if volatility == "rolling":
                    for rolling_window in np.arange(rolling_window_l, rolling_window_u, rolling_window_s):
                        self.run_trading_system(rolling_window=rolling_window,
                                                lower_threshold=lower_threshold)
                        mean_returns, std_returns, _, _, _ = self.get_returns_stats()
                        metric_value = self._evaluate_criteria(criteria, mean_returns, std_returns)
                        if metric_value > best_profit and self.get_num_trades() > min_trades:
                            best_profit = metric_value
                            best_lower_threshold = lower_threshold
                            best_rolling_window = rolling_window
                else:
                    self.run_trading_system(volatility=volatility,
                                            lower_threshold=lower_threshold)
                    mean_returns, std_returns, _, _, _ = self.get_returns_stats()
                    metric_value = self._evaluate_criteria(criteria, mean_returns, std_returns)
                    if metric_value > best_profit and self.get_num_trades() > min_trades:
                        best_profit = metric_value
                        best_lower_threshold = lower_threshold
                        best_rolling_window = None  # No rolling window for GARCH


        if verbose or plot:
            if volatility == "rolling":
                print(f"Best rolling window: {best_rolling_window}, Best lower threshold: {best_lower_threshold}")
            else:
                print(f"Best lower threshold: {best_lower_threshold:.2f}")
            self.run_trading_system(volatility= volatility, rolling_window=best_rolling_window, lower_threshold=best_lower_threshold, verbose=verbose, plot=plot)

        return best_profit, best_rolling_window, best_lower_threshold 

    

    def _evaluate_criteria(self, criteria, mean_returns, std_returns):
        if criteria == "profit":
            return self.profit
        elif criteria == "mean_returns":
            return mean_returns
        elif criteria == "sharpe":
            return mean_returns / std_returns if std_returns != 0 else 0
        return -float('inf')

       



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

