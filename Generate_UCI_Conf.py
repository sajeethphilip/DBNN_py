import requests
from bs4 import BeautifulSoup
import json
import os
import io
import sys
from ucimlrepo import fetch_ucirepo, list_available_datasets

def get_available_datasets():
    # Redirect stdout to capture the output
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    # Call the function that prints the datasets
    list_available_datasets()

    # Get the printed output and restore stdout
    output = new_stdout.getvalue()
    sys.stdout = old_stdout

    # Process the output to extract dataset names
    # Split by newlines and filter out empty lines
    lines = [line.strip() for line in output.split('\n') if line.strip()]

    # Extract dataset names (everything before the trailing spaces and numbers)
    datasets = []
    for line in lines:
        if line != "Available datasets:":  # Skip the header line
            # Split on whitespace and take everything except the last item (which is the number)
            name = ' '.join(line.split()[:-1]).strip()
            if name:  # Only add non-empty names
                datasets.append(name)

    return datasets

def fetch_dataset_info(dataset_name):
    try:
        dataset = fetch_ucirepo(name=dataset_name)

        # Determine the actual separator by checking the file extension and content
        file_url = dataset.metadata.data_url
        response = requests.get(file_url)
        if response.ok:
            first_line = response.text.split('\n')[0]
            if ',' in first_line:
                separator = ','
            elif ';' in first_line:
                separator = ';'
            elif '\t' in first_line:
                separator = '\t'
            else:
                separator = ' '
        else:
            separator = ','  # default if unable to determine

        # Handle target column - ensure it's a single value
        if isinstance(dataset.metadata.target_col, list):
            target = dataset.metadata.target_col[0]  # take first if multiple
        else:
            target = dataset.metadata.target_col

        dataset_info = {
            "name": dataset.metadata.name.lower(),
            "url": dataset.metadata.data_url,
            "columns": list(dataset.data.original.columns),
            "target": target,
            "instances": dataset.metadata.num_instances,
            "separator": separator,
            "has_header": True if dataset.metadata.feature_types else False
        }

        dataset_info["accuracy"] = 0.0  # placeholder for accuracy

        return dataset_info
    except Exception as e:
        print(f"Error fetching dataset {dataset_name}: {str(e)}")
        return None

def create_config_file(dataset_name, output_dir="."):
    dataset_info = fetch_dataset_info(dataset_name)

    if dataset_info is None:
        print(f"Could not fetch information for dataset '{dataset_name}'")
        return

    config = {
        "file_path": dataset_info["url"],
        "column_names": dataset_info["columns"],
        "target_column": dataset_info["target"],  # Now a single value
        "separator": dataset_info["separator"],   # Determined from actual file
        "has_header": dataset_info["has_header"],
         "likelihood_config": {
        "feature_group_size": 2,
        "max_combinations": 1000
    }
    }

    filename = f"{dataset_info['name'].lower().replace(' ', '_')}.conf"
    with open(os.path.join(output_dir, filename), 'w') as f:
        f.write(f"# {filename} best_accuracy: {dataset_info['accuracy']}, instances: {dataset_info['instances']}\n")
        json.dump(config, f, indent=4)
        print(f"Created configuration file: {filename}")


if __name__ == "__main__":
    # Get the list of available datasets
    available_datasets = get_available_datasets()
    print("Available datasets:")
    for i, dataset in enumerate(available_datasets, 1):
        print(f"{i}. {dataset}")

    dataset_name = input("\nEnter dataset name (press Enter to process all datasets): ")
    if dataset_name == '':
        # Process all datasets
        for dataset_name in available_datasets:
            print(f"\nProcessing dataset: {dataset_name}")
            create_config_file(dataset_name)
    else:
        # Process single dataset
        create_config_file(dataset_name)
