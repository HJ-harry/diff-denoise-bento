from PIL import Image
from pathlib import Path
import numpy as np

noise_level = 13.0
vol_name = "file1000033"
load_root = Path(f"./samples/knee_320_val_noise/gaussian/{noise_level}/{vol_name}")
save_root = Path(f"./samples/knee_320_val_noise_jpg/gaussian/{noise_level}/{vol_name}")
save_root.mkdir(exist_ok=True, parents=True)

f_list = sorted(list(load_root.glob("*.npy")))

for f in f_list:
    fname = str(f).split('/')[-1][:-4]
    img = np.clip(np.load(str(f)), 0.0, 1.0)
    img = Image.fromarray(np.uint8(img * 255.))
    
    save_fname = str(save_root / f"{fname}.jpg")
    img.save(save_fname)
    print(save_fname)