import cv2
import torch
import numpy as np
import pathlib
from pathlib import Path
import glob

# Patch PosixPath untuk Windows
pathlib.PosixPath = pathlib.WindowsPath

# Import YOLOv5 dependencies
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

# Load model
model_path = 'best.pt'
device = select_device('')
model = DetectMultiBackend(model_path, device=device, dnn=False)
stride, names = model.stride, model.names
#model.warmup(imgsz=(1, 3, 640, 640))  # optional

image = [x for x in glob.glob("D:/RISET/buoys.v1i.coco/train/*.jpg")]

# Untuk menyimpan titik deteksi Red_Ball
red_centers = []
green_centers = []
i = 0
while True:
    frame = cv2.imread(image[i])

    img = cv2.resize(frame, (640, 640))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).to(device)
    img_tensor = img_tensor.permute(2, 0, 1).float() / 255.0  # CHW, normalize
    img_tensor = img_tensor.unsqueeze(0)  # add batch dim

    with torch.no_grad():
        pred = model(img_tensor, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.45)

    red_centers.clear()
    green_centers.clear()
    
    matrix = 0
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], frame.shape).round()

            for *xyxy, conf, cls in det:
                class_name = names[int(cls)]
                x1, y1, x2, y2 = map(int, xyxy)
                rx, ry = (x1 + x2) // 2, (y1 + y2) // 2                
                gx, gy = (x1 + x2) // 2, (y1 + y2) // 2                
                if class_name == "red_buoy":
                    red_centers.append((rx, ry))
                    color = (0, 0, 255)
                    
                elif class_name == "green_buoy":
                    green_centers.append((gx, gy))
                    color = (0, 255, 0)
                else:
                    continue

                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{class_name}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Urutkan dari bawah ke atas (Y besar ke kecil)
    red_centers.sort(key=lambda pt: -pt[1])
    green_centers.sort(key=lambda pt: -pt[1])

    # Gambar jalur RED
    for i in range(len(red_centers) - 1):
        cv2.line(frame, red_centers[i], red_centers[i + 1], (0, 255, 0), 2)
        cv2.circle(frame, red_centers[i], 5, (0, 255, 0), -1)

    # Gambar jalur GREEN
    for i in range(len(green_centers) - 1):
        cv2.line(frame, green_centers[i], green_centers[i + 1], (255, 0, 0), 2)
        cv2.circle(frame, green_centers[i], 5, (255, 0, 0), -1)
    

    # Gambar titik tengah
    for i, j in zip(range(len(green_centers)-1), range(len(red_centers)-1)) : # Pastikan ada setidaknya satu bola merah dan hijau
       
        # Gambar garis antara titik tengah bola merah dan hijau
        cv2.line(frame, green_centers[i], red_centers[j], (255, 255, 0), 2)  # Warna cyan untuk garis

        # Hitung titik tengah dari garis
        mid_x = (red_centers[i][0] + green_centers[j][0]) // 2
        mid_y = (red_centers[i][1] + green_centers[j][1]) // 2
        mid_point = (mid_x, mid_y)

        # Gambar titik tengah
        cv2.circle(frame, mid_point, 5, (255, 255, 255), -1)
    

    
   

    # Titik akhir Red/Green
    if red_centers:
        cv2.circle(frame, red_centers[-1], 6, (0, 255, 255), -1)
    if green_centers:
        cv2.circle(frame, green_centers[-1], 6, (0, 255, 255), -1)

    cv2.imshow("Path Detection", frame)
    if cv2.waitKey(1000) & 0xFF == ord('q') & i == len(image):
        break
