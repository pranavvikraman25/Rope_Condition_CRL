import os

img_dir = "dataset/images"
mask_dir = "dataset/masks"

imgs = sorted(os.listdir(img_dir))
masks = sorted(os.listdir(mask_dir))

print(f"Images: {len(imgs)}, Masks: {len(masks)}")

if len(imgs) != len(masks):
    print("⚠️ Warning: Image and mask counts do not match!")

for img in imgs:
    name = os.path.splitext(img)[0]
    mask = os.path.join(mask_dir, name + ".png")
    if not os.path.exists(mask):
        print(f"❌ Missing mask for {img}")
    else:
        print(f"✅ Found: {img}")
