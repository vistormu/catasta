import gdown

url = "https://drive.google.com/drive/folders/1SC3QHO7WqwV3h2SIKGB65D3vU-m3bzL2?usp=sharing"
gdown.download_folder(url, use_cookies=False)
