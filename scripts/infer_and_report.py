# scripts/infer_and_report.py
import os, time, csv, sqlite3
from ultralytics import YOLO
import cv2

MODEL = "runs/detect/train/weights/best.pt"
model = YOLO(MODEL)
os.makedirs("results", exist_ok=True)

# Setup DB
conn = sqlite3.connect("results/rope_results.db")
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY,
    filename TEXT, timestamp TEXT, rust_count INTEGER, cut_count INTEGER,
    rust_area REAL, cut_area REAL, condition TEXT, action TEXT
)""")
conn.commit()

def map_action(rust_count, cut_count, rust_area_pct):
    # simple thresholds - adjust later from your Excel
    if cut_count >= 2:
        return "Critical","Replace immediately"
    if cut_count == 1:
        return "Single cut","Immediate inspection"
    if rust_area_pct > 0.2:
        return "Deeply rusted","Change rope immediately"
    if rust_area_pct > 0.08:
        return "Clearly rusted","Lubricate immediately"
    if rust_area_pct > 0.03:
        return "Rust between strands","Lubricate within 200k cycles"
    if rust_area_pct > 0.01:
        return "Rust between wires","Monitor"
    return "Well lubricated + No rust","No action required"

def process_image(img_path):
    results = model.predict(source=img_path, conf=0.3, verbose=False)
    r = results[0]
    img = cv2.imread(img_path)
    H,W = img.shape[:2]
    rust_count=0; cut_count=0; rust_area=0.0; cut_area=0.0
    for box,conf,cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
        x1,y1,x2,y2 = [int(v) for v in box.tolist()]
        area = (x2-x1)*(y2-y1) / (W*H)
        label = model.names[int(cls)]
        color = (0,0,255) if label=="rust" else (0,255,0)
        cv2.rectangle(img,(x1,y1),(x2,y2), color, 2)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,2)
        if label=="rust":
            rust_count+=1; rust_area+=area
        else:
            cut_count+=1; cut_area+=area
    rust_pct = rust_area
    cut_pct = cut_area
    condition, action = map_action(rust_count, cut_count, rust_pct)
    out = os.path.join("results","processed_"+os.path.basename(img_path))
    cv2.imwrite(out, img)
    # save DB
    c.execute("INSERT INTO results (filename,timestamp,rust_count,cut_count,rust_area,cut_area,condition,action) VALUES (?,?,?,?,?,?,?,?)",
              (os.path.basename(img_path), time.ctime(), rust_count, cut_count, rust_pct, cut_pct, condition, action))
    conn.commit()
    return out, condition, action, rust_count, cut_count, rust_pct, cut_pct

if __name__=="__main__":
    import glob
    rows=[]
    for im in glob.glob("dataset/images/val/*.jpg"):
        out, cond, act, rc, cc, rp, cp = process_image(im)
        rows.append([os.path.basename(im), rc, cc, rp, cp, cond, act])
    with open("results/report.csv","w",newline="") as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(['filename','rust_count','cut_count','rust_area_pct','cut_area_pct','condition','action'])
        writer.writerows(rows)
    print("Done. results in results/")
