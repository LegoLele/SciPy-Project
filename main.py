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

    return chat_df
    

# Define a file path
whatsapp_data_path = "Data/WhatsApp_data.txt"
df = read_data(whatsapp_data_path)
print(df)
