import csv

# Opens the WhatsApp dataset in the read mode
def read_csv():
    with open(whatsapp_data_path, "r") as file:
        read_dataset = csv.reader(file)
        for row in read_dataset:
            print(row)



# Define a file path
whatsapp_data_path = ""
read_csv()