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
    """
    Asks the user whether they wish to analyze their own exported data or use the default data.
    
    Returns:
        str or bool: If the user chooses to use their own data set ('Y' or 'y'), the function prompts for the file path and returns the path (str).
                     If the user chooses not to use their own data set ('N' or 'n'), the function returns False to use the default data.
    """

    user_choice = input('Do you wish to analyze your own exported data set? (Y/N)\n')
    if user_choice.lower() == 'n':
        return False
    elif user_choice.lower() == 'y':
        while True:
            user_input = input('Please make sure you put the raw txt export of the chat to avoid issues with the code.\nWhat\'s the path of your exported chat file?\n')
            if os.path.isfile(user_input):
                return user_input
            else:
                print("Invalid path! Please try again!\n")

if __name__ == "__main__":
   # Default file path
    data_path = "Data/WhatsApp_data.txt"

    custom_data = ask_for_custom_data()
    if custom_data:
        data_path = custom_data

    df = helper.read_data(data_path)

    print(df)
    helper.text_analysis(df)
    helper.activity_heatmap(df)
    
    print(mp.predict_chat_messages(df,20))
