import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA


def read_data(whatsapp_data_path):
    """
    Reads the data, extracts timestamps, senders and messages, converts the data into dataframe.

    Args:
        whatsapp_data_path (str): The file path of the WhatsApp chat data.
        
    Returns:
        chat_df: The WhatsApp chat data is returned as a dataframe.
    """ 

    # Captures timestamp, sender and message based on the dataset
    regex = r"\[(.*?)\]\s(.*?):\s(.*)"

    # Opens the WhatsApp dataset in the read mode
    with open(whatsapp_data_path, "r", encoding = "UTF-8") as file:
        txt_reader = file.read()

    # Finds all regex matches in the text file
    reg_match = re.findall(regex, txt_reader)
    
    # Each message is stored as dictionaries
    chat_df = []
    for i in range(len(reg_match)):
        j = {
        "Time": reg_match[i][0],
        "Sender": reg_match[i][1],
        "Message": reg_match[i][2]
        }

        # Dictionaries are appended to the chat_df list
        chat_df.append(j)

    # chat_df list is converted into a dataframe
    chat_df = pd.DataFrame(chat_df)

    # Deletes first two rows (meassages not sent by a human but by WhatsApp)
    chat_df = chat_df[2:]

    return chat_df

def is_emoji(char):

    # Unicode ranges for emojis
    emojis = [
        (0x1F600, 0x1F64F),  # Emoticons
        (0x1F300, 0x1F5FF),  # Miscellaneous Symbols and Pictographs
        (0x1F910, 0x1F95F),  # Clothing and Accessories
        (0x1F980, 0x1F9EF),  # Food, Drink, and Cooking
        (0x1F680, 0x1F6FF),  # Transport and Map Symbols
        (0x1F700, 0x1F77F),  # Alchemical Symbols
        (0x1F780, 0x1F7FF),  # Geometric Shapes Extended
        (0x1F900, 0x1F9FF),  # Supplemental Symbols and Pictographs
    ]

    for start, end in emojis:
        if start <= ord(char) <= end:
            return True
    return False

def text_analysis(df):
    # Most active time
    df['Time'] = pd.to_datetime(df['Time'], format='%d.%m.%y, %H:%M:%S')
    df['Hour'] = df['Time'].dt.hour
    hourly_counts = df.groupby('Hour').size()
    max_hour = hourly_counts.idxmax()
    max_messages = hourly_counts[max_hour]
    print(f"The most messages were sent between {max_hour} and {(max_hour+1)} o'clock ({max_messages} messages).")

    # Least active time
    min_hour = hourly_counts.idxmin()
    min_messages = hourly_counts[min_hour]
    print(f"Excluding hours without meassages, the least messages were sent between {min_hour} and {(min_hour+1)} o'clock ({min_messages} messages).")

    # Most active author
    sender_counts = df['Sender'].value_counts()
    max_sender = sender_counts.idxmax()
    max_messages = sender_counts[max_sender]
    max_messages_percent = round(max_messages/len(df) * 100, 2)
    print(f"The sender with the most messages is {max_sender} with {max_messages} messages ({max_messages_percent}%).")

    # Least active author
    min_sender = sender_counts.idxmin()
    min_messages = sender_counts[min_sender]
    min_messages_percent = round(min_messages/len(df) * 100, 2)
    print(f"The sender with the least messages is {min_sender} with {min_messages} messages ({min_messages_percent}%).")

    # Total number of messages
    total_messages = len(df)
    print(f"The total number of messages: {total_messages}")

    # Most frequently used emoji
    emoji_list = []
    for message in df["Message"]:
        emoji_list.extend([e for e in message if is_emoji(e)])

    count_emojis = pd.Series(emoji_list).value_counts()
    if not count_emojis.empty:
        most_freq_emoji = count_emojis.idxmax()
        most_freq_emoji_count = count_emojis[most_freq_emoji]
        print(f"The most frequently used emoji is {most_freq_emoji} which occured {most_freq_emoji_count} times.")

def activity_heatmap():
        user_heatmap_data = df.groupby(["Sender", "Hour"]).size().reset_index(name = "message_count")

        heatmap = user_heatmap_data.pivot(index = "Sender", columns = "Hour", values = "message_count")
        plt.figure(figsize =(10, 5))
        purple_cmap = sns.color_palette("flare", as_cmap=True)
        sns.heatmap(heatmap, cmap=purple_cmap, annot=True, fmt=".0f")
        plt.title("WhatsApp User Activity on Hourly Basis")
        plt.xlabel("Hour of the Day")
        plt.ylabel("Sender")
        plt.tight_layout()
        plt.show()

def prepare_data(df):
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
    # Create time index for the data
    time_index = pd.date_range(start=data.index[0], periods=len(data), freq=data.index.inferred_freq)

    # Fit ARIMA model using the time index
    model = ARIMA(data, order=(1, 0, 1), dates=time_index)  # Specify the dates parameter
    model_fit = model.fit()

    # Summary of the model
    print(model_fit.summary())

    return model_fit

def make_predictions(model_fit, n_periods):
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
    # Prepare data
    df_arima = prepare_data(df)

    # Check stationarity
    is_stationary = check_stationarity(df_arima['MessagesCount'])

    if not is_stationary:
        print("Data is not stationary. Apply transformations to make it stationary.")
        return

    # Fit ARIMA model
    model_fit = fit_arima_model(df_arima)

    # Make predictions
    forecast_df = make_predictions(model_fit, n_periods)

    return forecast_df

if __name__ == "__main__":
    # Define a file path
    whatsapp_data_path = "Data/WhatsApp_data.txt"
    df = read_data(whatsapp_data_path)

    print(df)
    text_analysis(df)
    activity_heatmap()
    print(predict_chat_messages(df,20))
