import gdown
import os


def download_file_from_google_drive(id, dir, file):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f"Created directory at {dir}")
    else:
        print(f"Directory already exists at {dir}")

    file_path = os.path.join(dir, file)
    if not os.path.exists(file_path):
        print(f"File does not exist at {file_path}, downloading now...")
        gdown.download_folder(id=id, output=dir)
        print(f"Downloaded file from Google Drive to {file_path}")
    else:
        print(f"File already exists at {file_path}, skipping download.")
