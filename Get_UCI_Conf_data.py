import json
import requests
import os
import io
import pandas as pd

def download_uci_dataset(config_path):
    # Read the configuration file
    with open(config_path, 'r') as f:
        # Skip the first line with accuracy info
        next(f)
        config = json.load(f)

    # Extract file info from config
    file_url = config['file_path']
    dataset_name = os.path.basename(config_path).replace('.conf', '.csv')
    separator = config['separator']
    has_header = config['has_header']

    try:
        # Download the data
        response = requests.get(file_url)
        response.raise_for_status()

        # Read the data into a pandas DataFrame
        if has_header:
            df = pd.read_csv(io.StringIO(response.text), sep=separator)
        else:
            df = pd.read_csv(io.StringIO(response.text), sep=separator,
                           names=config['column_names'])

        # Save to local CSV file
        df.to_csv(dataset_name, index=False)
        print(f"Successfully downloaded dataset to {dataset_name}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the dataset: {str(e)}")
    except pd.errors.EmptyDataError:
        print("The downloaded file is empty")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    config_dir = "UCI_Configs"

    # List all configuration files
    config_files = [f for f in os.listdir(config_dir)
                   if f.endswith('.conf')]

    print("Available configuration files:")
    for i, conf in enumerate(config_files, 1):
        print(f"{i}. {conf}")

    choice = input("\nEnter config file name (press Enter for all): ")

    if choice == '':
        # Process all configuration files
        for conf in config_files:
            config_path = os.path.join(config_dir, conf)
            print(f"\nProcessing: {conf}")
            download_uci_dataset(config_path)
    else:
        # Process single configuration file
        config_path = os.path.join(config_dir, choice)
        if os.path.exists(config_path):
            download_uci_dataset(config_path)
        else:
            print("Configuration file not found")
