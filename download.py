import os
from zipfile import ZipFile
import glob
import requests
CHECKPOINT_URL = r"https://www.dropbox.com/scl/fi/bicrtoufvj5zxc0jqy4cj/yolov8-models-kerascv.zip?rlkey=majuikz2z9oofchbkw5uw4amu&dl=1"
def download_and_unzip(url, save_path):

    print("Downloading and extracting assets...", end="")
    file = requests.get(url)
    open(save_path, "wb").write(file.content)

    try:
        # Extract zipfile.
        if save_path.endswith(".zip"):
            with ZipFile(save_path) as zip:
                zip.extractall(os.path.split(save_path)[0])

        print("Done")
    except:
        print("Invalid file")

CHECKPOINT_DIR = "yolov8-models-kerascv"
CKPT_ZIP_PATH = os.path.join(os.getcwd(), f"{CHECKPOINT_DIR}.zip")
if not os.path.exists(CHECKPOINT_DIR):
    download_and_unzip(CHECKPOINT_URL, CKPT_ZIP_PATH)
    os.remove(CKPT_ZIP_PATH)