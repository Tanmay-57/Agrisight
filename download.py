import os
from zipfile import ZipFile
import glob
import requests
SAVED_MODEL_URL = r"https://www.dropbox.com/scl/fi/38qbm2kd9jq4gfw3qo8u1/yolov8-models-kerascv.zip?rlkey=g3h0qj270u4n1q7ri63kkeh9p&st=0vfcmuhu&dl=1"

def download_and_unzip(url, save_path):
    print("Downloading and extracting assets...", end="")
    file = requests.get(url)
    open(save_path, "wb").write(file.content)

    try:
        if save_path.endswith(".zip"):
            with ZipFile(save_path) as zip:
                zip.extractall(os.path.split(save_path)[0])

        print("Done")
    except:
        print("Invalid file")


CHECKPOINT_DIR = "yolov8-models-kerascv"
SAVED_MODEL_ZIP = os.path.join(os.getcwd(), f"{CHECKPOINT_DIR}.zip")
if not os.path.exists(CHECKPOINT_DIR):
    download_and_unzip(SAVED_MODEL_URL, SAVED_MODEL_ZIP)
    os.remove(SAVED_MODEL_ZIP)
