import os
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path

base_path = Path("dataset")  # Folder containing images/train and labels/train
image_dir = base_path / "images" / "train"
label_dir = base_path / "labels" / "train"
val_ratio = 0.1
random_state = 42


for subfolder in ["images/val", "labels/val"]:
    os.makedirs(base_path / subfolder, exist_ok=True)

image_files = sorted([f for f in image_dir.glob("*.jpg")])
label_files = [label_dir / (img.stem + ".txt") for img in image_files]

train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
    image_files, label_files, test_size=val_ratio, random_state=random_state
)

def move_files(files, destination_dir):
    for f in files:
        shutil.move(str(f), base_path / destination_dir / f.name)

move_files(val_imgs, "images/val")
move_files(val_lbls, "labels/val")

print(f"Split into {len(train_imgs)} train and {len(val_imgs)} val images.")
