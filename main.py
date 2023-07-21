import pandas as pd
import re

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
    
def simple_analysis(df):
    # Most active time
    df['Time'] = pd.to_datetime(df['Time'], format='%d.%m.%y, %H:%M:%S')
    df['hour'] = df['Time'].dt.hour
    hourly_counts = df.groupby('hour').size()
    max_hour = hourly_counts.idxmax()
    max_messages = hourly_counts[max_hour]
    print(f"The most messages were sent between {max_hour} and {(max_hour+1)} o'clock ({max_messages} messages).")

    # Least active time
    min_hour = hourly_counts.idxmin()
    min_messages = hourly_counts[min_hour]
    print(f"The least messages were sent between {min_hour} and {(min_hour+1)} o'clock ({min_messages} messages).")

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

if __name__ == "__main__":
    # Define a file path
    whatsapp_data_path = "Data/WhatsApp_data.txt"

    df = read_data(whatsapp_data_path)
    print(df)
    simple_analysis(df)
