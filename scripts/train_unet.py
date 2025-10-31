import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm


def iou_score(pred, target, num_classes=3):
    ious = []
    for cls in range(num_classes):
        inter = ((pred == cls) & (target == cls)).sum()
        union = ((pred == cls) | (target == cls)).sum()
        if union == 0:
            ious.append(1.0)
        else:
            ious.append(inter / union)
    return np.mean(ious)


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



class RopeDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir

        # Make extension handling flexible (.jpg, .JPG, .png, .PNG)
        self.images = [f for f in os.listdir(img_dir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        self.transform_img = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])


        self.transform_mask = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.NEAREST),
            transforms.PILToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        base = os.path.splitext(img_name)[0]

        possible_exts = [".png", ".PNG", ".jpg", ".JPG"]
        mask_path = None
        for ext in possible_exts:
            candidate = os.path.join(self.mask_dir, base + ext)
            if os.path.exists(candidate):
                mask_path = candidate
                break
        if mask_path is None:
            raise FileNotFoundError(f"No mask found for {base}")

        img = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")
        mask = Image.open(mask_path)

        img = self.transform_img(img)
        mask = self.transform_mask(mask).squeeze(0).long()

        return img, mask



dataset = RopeDataset("dataset/images", "dataset/masks")
print("Total training images found:", len(dataset))
if len(dataset) == 0:
    raise ValueError("❌ No valid image–mask pairs found. Check dataset paths and extensions.")

loader = DataLoader(dataset, batch_size=2, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


num_epochs = 52
os.makedirs("models", exist_ok=True)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for img, mask in tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=80):
        img, mask = img.to(device), mask.to(device)
        outputs = model(img)
        loss = criterion(outputs, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()



        
# --- IoU Evaluation ---
    model.eval()
    with torch.no_grad():
        total_iou = 0
        count = 0
        for img, mask in loader:
            img, mask = img.to(device), mask.to(device)
            output = model(img)
            pred = torch.argmax(output, dim=1).cpu().numpy()
            true = mask.cpu().numpy()
            total_iou += iou_score(pred, true, num_classes=3)
            count += 1
        avg_iou = total_iou / count

    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss/len(loader):.4f} - IoU: {avg_iou:.4f}")

        

    avg_loss = running_loss / len(loader)
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
    
if (epoch + 1) % 10 == 0:
    torch.save(model.state_dict(), f"models/unet_epoch_{epoch+1}.pth")

torch.save(model.state_dict(), "models/unet_model.pth")
print("✅ Training complete. Model saved as models/unet_model.pth")
