# Opens the WhatsApp dataset in the read mode
def read_data():
    with open(whatsapp_data_path, "r") as file:
        reader = file.read()
        print(reader)
        

# Define a file path
whatsapp_data_path = "Data/WhatsApp_data.txt"
read_data()