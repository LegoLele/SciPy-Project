import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

def prepare_data(df):
    """
    Prepare the DataFrame containing the group chat data for ARIMA.

    Returns:
        pandas.DataFrame: A new DataFrame with messages count resampled per hour, indexed by time.
    """

    # Convert 'Time' column to datetime and set it as the index
    df['Time'] = pd.to_datetime(df['Time'], format='%d.%m.%y, %H:%M:%S')
    df.set_index('Time', inplace=True)

    # Resample data to get messages per hour
    messages_per_hour = df.resample('H').count()
    messages_per_hour.rename(columns={'Message': 'MessagesCount'}, inplace=True)

    # Drop unnessecary columns
    messages_per_hour.drop(columns='Sender', inplace=True)

    # Return DataFrame with messages per hour
    return messages_per_hour

def check_stationarity(data):
    """
    Check stationarity of the group chat data using the Dickey-Fuller test. (Data needs to be stationary for ARIMA)

    Returns:
        bool: True if the data is stationary (reject the null hypothesis), False if it is non-stationary (fail to reject the null hypothesis).
    """

    # Perform Dickey-Fuller test
    result = adfuller(data, autolag='AIC')

    # Print test results
    print("Dickey-Fuller Test Results:")
    print(f'Test Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print(f'Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')

    # Check if p-value is below threshhold (normally 0.05)
    if result[1] <= 0.05:
        print("Data is stationary (reject the null hypothesis).")
        return True
    else:
        print("Data is non-stationary (fail to reject the null hypothesis).")
        return False

def fit_arima_model(data):
    """
    Fit an ARIMA model to the given time series data.

    Returns:
        model_fit (statsmodels.tsa.arima.model.ARIMAResults): The fitted ARIMA model results.
    """

    # Create time index for the data
    time_index = pd.date_range(start=data.index[0], periods=len(data), freq=data.index.inferred_freq)

    # Fit ARIMA model using the time index
    model = ARIMA(data, order=(1, 0, 1), dates=time_index)  # Specify the dates parameter
    model_fit = model.fit()

    # Summary of the model
    print(model_fit.summary())

    return model_fit

def make_predictions(model_fit, n_periods):
    """
    Generate forecasts for a time series model.

    Returns:
        forecast_df (DataFrame): A pandas DataFrame containing the forecasted values with a time index.
    """
    # Forecast n_periods into the future
    forecast = model_fit.forecast(steps=n_periods)

    # Get the last time point in the original data
    last_time = model_fit.model.nobs - 1
    last_time_point = model_fit.model.data.dates[last_time]

    # Create a time index for the forecast
    forecast_index = pd.date_range(start=last_time_point, periods=n_periods, freq=model_fit.model.data.freq)

    # Create a DataFrame with the forecast
    forecast_df = pd.DataFrame(data=forecast, index=forecast_index, columns=['Forecast'])

    return forecast_df

def predict_chat_messages(df, n_periods):
    """
    Predicts future chat message counts using the ARIMA model.

    Returns:
    pandas.DataFrame or None: A DataFrame containing the forecasted chat message counts for the specified
                              number of future periods. Returns None if the data is not stationary and
                              cannot be fitted with an ARIMA model.
    """
    
    # Prepare data
    df_arima = prepare_data(df)

    # Check stationarity
    is_stationary = check_stationarity(df_arima['MessagesCount'])

    if not is_stationary:
        print("Data is not stationary. Apply transformations to make it stationary.")
        return None

    # Fit ARIMA model
    model_fit = fit_arima_model(df_arima)

    # Make predictions
    forecast_df = make_predictions(model_fit, n_periods)

    return forecast_df