import os
import zipfile
import requests
from pathlib import Path

def download_plantdoc_dataset(data_dir='data'):
    """
    Download the PlantDoc dataset from GitHub.
    The dataset is available at: https://github.com/pratikkayal/PlantDoc-Dataset
    """
    url = "https://github.com/pratikkayal/PlantDoc-Dataset/archive/refs/heads/master.zip"
    zip_path = os.path.join(data_dir, 'plantdoc.zip')
    extract_path = os.path.join(data_dir, 'plantdoc')

    # Create data directory if it doesn't exist
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    print("Downloading PlantDoc dataset...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

        # Rename extracted folder
        extracted_folder = os.path.join(data_dir, 'PlantDoc-Dataset-master')
        if os.path.exists(extracted_folder):
            os.rename(extracted_folder, extract_path)

        # Clean up zip file
        os.remove(zip_path)

        print(f"Dataset downloaded and extracted to {extract_path}")
        return extract_path

    except Exception as e:
        print(f"Download failed: {e}")
        print("Please manually download the PlantDoc dataset and place it in the 'data/plantdoc' directory.")
        print("You can find it at: https://github.com/pratikkayal/PlantDoc-Dataset")
        return None

if __name__ == "__main__":
    download_plantdoc_dataset()
