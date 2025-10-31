# scripts/masks_to_bboxes.py
import os, cv2
from glob import glob
img_dir = "dataset/images"
mask_dir = "dataset/masks"
out_dir = "dataset/labels"
os.makedirs(out_dir, exist_ok=True)

for mask_path in glob(mask_dir+"/*.png"):
    base = os.path.basename(mask_path).rsplit(".",1)[0]
    img_path = os.path.join(img_dir, base + ".jpg")
    if not os.path.exists(img_path):
        continue
    img = cv2.imread(img_path); h,w = img.shape[:2]
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    lines=[]
    # map: mask value 1->rust (class 0), 2->cut (class 1); change if needed
    for cls_val, cls_idx in [(1,0),(2,1)]:
        binmask = (mask==cls_val).astype('uint8')*255
        cnts,_ = cv2.findContours(binmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x,y,ww,hh = cv2.boundingRect(c)
            if ww*hh < 50: continue
            xc = (x + ww/2)/w
            yc = (y + hh/2)/h
            wn = ww/w
            hn = hh/h
            lines.append(f"{cls_idx} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
    open(os.path.join(out_dir, base+".txt"), "w").write("\n".join(lines))
print("Done masks->YOLO labels")
