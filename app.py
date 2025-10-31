import streamlit as st
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import io


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

st.set_page_config(page_title="Rope Condition Detector", layout="wide")
st.title("🪜 Rope Condition Detection System (Rust / Cut Analysis)")
st.markdown("Upload one or more rope images to detect **rust** and **wire cuts** automatically.")


uploaded_files = st.file_uploader(
    "Upload rope image(s)", type=["jpg", "png"], accept_multiple_files=True
)

if uploaded_files:
    # Load Model Once
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    model.load_state_dict(torch.load("models/unet_model.pth", map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Output folder
    os.makedirs("results", exist_ok=True)

    # Process each uploaded file
    for file in uploaded_files:
        st.subheader(f"🧾 Analyzing: {file.name}")
        img = Image.open(file).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        img_np = np.array(img.resize((256, 256)))
        output_vis = img_np.copy()
        classes = {1: "Rust", 2: "Cut"}
        colors = {1: (0, 0, 255), 2: (0, 255, 0)}
        counts = {1: 0, 2: 0}

        for cls, name in classes.items():
            mask = (prediction == cls).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 50:  # Ignore small noise
                    x, y, w, h = cv2.boundingRect(cnt)
                    counts[cls] += 1
                    label = f"{name}{counts[cls]}"
                    cv2.rectangle(output_vis, (x, y), (x+w, y+h), colors[cls], 2)
                    cv2.putText(output_vis, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[cls], 2)

        # Save output
        result_path = os.path.join("results", f"boxed_{file.name}")
        cv2.imwrite(result_path, cv2.cvtColor(output_vis, cv2.COLOR_RGB2BGR))

        # Display image
        st.image(output_vis, caption=f"Detected Rust: {counts[1]}, Cuts: {counts[2]}", use_column_width=True)

        # ---------------------------------------------------------
        # 🧾 Health Summary Logic
        # ---------------------------------------------------------
        rust_count, cut_count = counts[1], counts[2]
        if cut_count >= 2 or rust_count >= 3:
            condition = "🔴 Unsafe – Replacement Required"
        elif cut_count == 1 or rust_count == 1 or rust_count == 2:
            condition = "🟠 Moderate – Maintenance Recommended"
        else:
            condition = "🟢 Healthy – No Issues Detected"

        st.markdown(f"### 🧩 Rope Health Summary")
        st.markdown(f"**Rust Zones:** {rust_count}  |  **Cut Zones:** {cut_count}")
        st.markdown(f"**Condition:** {condition}")

        # ---------------------------------------------------------
        # 📥 Download Button
        # ---------------------------------------------------------
        result_pil = Image.fromarray(output_vis)
        buf = io.BytesIO()
        result_pil.save(buf, format="JPEG")
        byte_im = buf.getvalue()

        st.download_button(
            label="📥 Download Processed Image",
            data=byte_im,
            file_name=f"processed_{file.name}",
            mime="image/jpeg"
        )

        st.divider()
