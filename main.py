import pandas as pd
import re
from datetime import datetime


# Opens the WhatsApp dataset in the read mode
def read_data():
    regex = r"\[(.*?)\]\s(.*?):\s(.*)"
    with open(whatsapp_data_path, "r", encoding = "UTF-8") as file:
        txt_reader = file.read()
        #print(line_reader)
    reg_match = re.findall(regex, txt_reader)
        
    chat_df = []
    for i in range(len(reg_match)):
        j = {
        "Time": reg_match[i][0],
        "Sender": reg_match[i][1],
        "Message": reg_match[i][2]
        }

        chat_df.append(j)

    chat_df = pd.DataFrame(chat_df)

    return chat_df
    

# Define a file path
whatsapp_data_path = "Data/WhatsApp_data.txt"
df = read_data()
print(df)
