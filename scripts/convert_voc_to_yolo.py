import os
import xml.etree.ElementTree as ET
import cv2

print("Script started")

def convert_box(size, xmin, ymin, xmax, ymax):
    dw = 1.0 / size[1]
    dh = 1.0 / size[0]

    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin

    return (
        x_center * dw,
        y_center * dh,
        w * dw,
        h * dh
    )

def process_folder(img_dir, xml_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    for file in os.listdir(xml_dir):
        if not file.endswith(".xml"):
            continue

        tree = ET.parse(os.path.join(xml_dir, file))
        root = tree.getroot()

        filename = root.find("filename").text
        img_path = os.path.join(img_dir, filename)

        img = cv2.imread(img_path)
        if img is None:
            print("Image not found:", filename)
            continue

        h, w = img.shape[:2]

        yolo_lines = []

        for obj in root.findall("object"):
            xmin = float(obj.find("bndbox/xmin").text)
            ymin = float(obj.find("bndbox/ymin").text)
            xmax = float(obj.find("bndbox/xmax").text)
            ymax = float(obj.find("bndbox/ymax").text)

            x, y, bw, bh = convert_box((h, w), xmin, ymin, xmax, ymax)
            yolo_lines.append(f"0 {x} {y} {bw} {bh}")

        txt_name = filename.replace(".jpg", ".txt")
        with open(os.path.join(out_dir, txt_name), "w") as f:
            f.write("\n".join(yolo_lines))


base = "."

for split in ["train", "val", "test"]:
    print("Processing:", split)   # 👈 ADD THIS LINE
    process_folder(
        f"{base}/{split}/img",
        f"{base}/{split}/xml",
        f"{base}/{split}/labels"
    )