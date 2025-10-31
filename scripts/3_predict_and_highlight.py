import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

# ---------------------------------------------------------
# 🧠 U-Net model (same as used during training)
# ---------------------------------------------------------
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 2, stride=2)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ---------------------------------------------------------
# ⚙️ Load model
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model.load_state_dict(torch.load("models/unet_model.pth", map_location=device))
model.eval()

# ---------------------------------------------------------
# 📸 Choose an image for testing
# ---------------------------------------------------------
test_img_path = "dataset/images/PA180001.jpg"  # or any other image
img = Image.open(test_img_path).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
input_tensor = transform(img).unsqueeze(0).to(device)

# ---------------------------------------------------------
# 🔍 Predict mask
# ---------------------------------------------------------
with torch.no_grad():
    output = model(input_tensor)
prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

# ---------------------------------------------------------
# 🎨 Convert prediction to color + bounding boxes
# ---------------------------------------------------------
img_np = np.array(img.resize((256, 256)))
output_vis = img_np.copy()

# Define classes
classes = {1: "Rust", 2: "Cut"}
colors = {1: (0, 0, 255), 2: (0, 255, 0)}  # Rust=Red, Cut=Green
counts = {1: 0, 2: 0}

for cls, name in classes.items():
    mask = (prediction == cls).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50:  # ignore very small noise areas
            x, y, w, h = cv2.boundingRect(cnt)
            counts[cls] += 1
            label = f"{name}{counts[cls]}"
            cv2.rectangle(output_vis, (x, y), (x+w, y+h), colors[cls], 2)
            cv2.putText(output_vis, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, colors[cls], 2)

# ---------------------------------------------------------
# 💾 Save / Show
# ---------------------------------------------------------
os.makedirs("results", exist_ok=True)
save_path = os.path.join("results", f"boxed_{os.path.basename(test_img_path)}")
cv2.imwrite(save_path, cv2.cvtColor(output_vis, cv2.COLOR_RGB2BGR))

print("✅ Detection complete.")
print(f"Saved output image with boxes: {save_path}")
print(f"Detected: {counts[1]} rust zones, {counts[2]} cut zones")
