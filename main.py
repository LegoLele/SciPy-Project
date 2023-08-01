import helper
import message_prediction as mp
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

def ask_for_custom_data():
    user_choice = input('Do you wish to analize your own exported data set? (Y/N)\n')
    if user_choice == 'N' or 'n':
        return False
    elif user_choice == 'Y' or 'y':
        return True
    
    while True:
        user_choice = input('Incorrect input!\nDo you wish to analize your own exported data set? (Y/N)\n')
        if user_choice == 'N' or user_choice =='n':
            return False
        if user_choice == 'Y' or user_choice == 'y':
            return True

if __name__ == "__main__":
    # File path for future use
    data_path = ''

    custom_data = ask_for_custom_data()
    if custom_data:
        user_input = input('Please make sure you put the raw txt export of the chat to avoid issues with the code.\nWhat\'s the path of your exported chat file?\n')
        if os.path.isfile(user_input):
            data_path = user_input
        else:
            while True:
                user_input = input('Invalid path!\nWhat\'s the path of your exported chat file?\n')
                if os.path.isfile(user_input):
                    data_path = user_input
                    break
    else:
        data_path = "Data/WhatsApp_data.txt"

    df = helper.read_data(data_path)

    print(df)
    helper.text_analysis(df)
    helper.activity_heatmap(df)
    
    print(mp.predict_chat_messages(df,20))
