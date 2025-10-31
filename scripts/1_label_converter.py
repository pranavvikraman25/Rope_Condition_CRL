import labelme
import os
import numpy as np
from PIL import Image

input_dir = "dataset/annotations"
output_dir = "dataset/masks"
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if file.endswith(".json"):
        path = os.path.join(input_dir, file)
        data = labelme.LabelFile(filename=path)
        img = labelme.utils.img_data_to_arr(data.imageData)
        lbl, _ = labelme.utils.shapes_to_label(
            img.shape, data.shapes, label_name_to_value={"_background_": 0, "rust": 1, "cut": 2}
        )
        Image.fromarray(lbl.astype(np.uint8)).save(
            os.path.join(output_dir, file.replace(".json", ".png"))
        )

print("✅ Masks created successfully.")
