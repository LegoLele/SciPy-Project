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
    input = input('Do you wish to analize your own exported data set? (Y/N)\n')
    if input == 'N' or 'n':
        return False
    if input == 'Y' or 'y':
        return True
    
    while True:
        input = input('Incorrect input!\nDo you wish to analize your own exported data set? (Y/N)\n')
        if input == 'N' or 'n':
            return False
        if input == 'Y' or 'y':
            return True

if __name__ == "__main__":
    # File path for future use
    data_path = ''

    custom_data = ask_for_custom_data()
    if custom_data:
        input = input('Please make sure you put the raw txt export of the chat to avoid issues with the code.\nWhat\'s the path of your exported chat file?\n')
        if os.path.isfile(input):
            data_path = input
        else:
            while True:
                input = input('Invalid path!\nWhat\'s the path of your exported chat file?\n')
                if os.path.isfile(input):
                    data_path = input
                    break
    else:
        data_path = "Data/WhatsApp_data.txt"

    df = helper.read_data(data_path)

    print(df)
    helper.text_analysis(df)
    helper.activity_heatmap(df)
    
    print(mp.predict_chat_messages(df,20))
