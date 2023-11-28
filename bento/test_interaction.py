import requests
import numpy as np
from pathlib import Path

noise_type = "gaussian"
std = 13.0
load_root_input = Path(f'./samples/knee_320_val_noise_jpg/{noise_type}/{std}')
input_list = sorted(list(load_root_input.glob('*/*.jpg')))
input_fname = str(input_list[0])


response = requests.post(
    "http://0.0.0.0:3000/denoise",
   files = {"upload_file": open(str(input_fname), 'rb')},
   headers= {"content-type": "multipart/form-data"}
)

print(response.shape)