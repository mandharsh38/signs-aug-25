import os
import json
import shutil
import argparse
from tqdm import tqdm  # 🆕 Progress bar support

# Argument parser
parser = argparse.ArgumentParser(description="Convert LabelMe JSONs to YOLO segmentation format.")
parser.add_argument('--input-folder', required=True, help='Folder with JSON and JPG files')
args = parser.parse_args()

input_folder = args.input_folder
output_labels_folder = "dataset/labels/train/"
output_images_folder = "dataset/images/train/"
classes_file = "classes.txt"
data_yaml_file = "data.yaml"

# Ensure output directories exist
os.makedirs(output_labels_folder, exist_ok=True)
os.makedirs(output_images_folder, exist_ok=True)

# 🔍 Step 1: Collect unique labels
print("📦 Scanning JSONs to generate classes.txt...")
unique_labels = set()
json_files = [f for f in os.listdir(input_folder) if f.endswith(".json")]

for filename in tqdm(json_files, desc="Scanning labels"):
    try:
        with open(os.path.join(input_folder, filename), "r") as f:
            data = json.load(f)
            for shape in data.get("shapes", []):
                label = shape.get("label")
                if label:
                    unique_labels.add(label)
    except Exception as e:
        print(f"❌ Error reading {filename}: {e}")

# Save sorted labels to classes.txt
sorted_labels = sorted(unique_labels)
with open(classes_file, "w") as f:
    for label in sorted_labels:
        f.write(label + "\n")
print(f"✅ Generated classes.txt with {len(sorted_labels)} classes.")

# Reload class list
class_list = sorted_labels

# 🔄 Step 2: Convert annotations
print("🧾 Converting annotations to YOLO format...")
for filename in tqdm(json_files, desc="Converting"):
    json_path = os.path.join(input_folder, filename)
    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        image_width = data.get("imageWidth")
        image_height = data.get("imageHeight")

        yolo_lines = []

        for shape in data.get("shapes", []):
            label = shape.get("label")
            if not label or label not in class_list:
                continue

            if shape.get("shape_type") != "polygon":
                continue  # Only polygons for YOLO segmentation

            class_id = class_list.index(label)
            points = shape.get("points", [])

            normalized_points = []
            for x, y in points:
                x_norm = round(x / image_width, 6)
                y_norm = round(y / image_height, 6)
                normalized_points.extend([x_norm, y_norm])

            yolo_line = f"{class_id} " + " ".join(map(str, normalized_points))
            yolo_lines.append(yolo_line)

        # Write YOLO txt label if valid annotations found
        if yolo_lines:
            base_name = os.path.splitext(filename)[0]
            label_output_path = os.path.join(output_labels_folder, base_name + ".txt")
            with open(label_output_path, "w") as f:
                f.write("\n".join(yolo_lines))

            # Copy image
            image_file = base_name + ".jpg"
            image_src_path = os.path.join(input_folder, image_file)
            image_dst_path = os.path.join(output_images_folder, image_file)
            if os.path.exists(image_src_path):
                shutil.copy(image_src_path, image_dst_path)
            else:
                print(f"⚠️ Image not found: {image_file}")

    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")

# 📄 Step 3: Generate data.yaml
print("📝 Generating data.yaml...")
with open(data_yaml_file, "w") as f:
    f.write("train: dataset/images/train\n")
    f.write("val: dataset/images/train  # Same as train for now\n\n")
    f.write(f"nc: {len(class_list)}\n")
    f.write("names: [")
    f.write(", ".join([f"'{name}'" for name in class_list]))
    f.write("]\n")

print(f"✅ data.yaml generated with {len(class_list)} classes.")
