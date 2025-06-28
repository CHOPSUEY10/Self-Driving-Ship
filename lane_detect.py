import glob
import pickle
import time
import cv2 as cv
import numpy as np
import serial
import torch
import pathlib
from pathlib import Path

# Patch PosixPath untuk Windows
pathlib.PosixPath = pathlib.WindowsPath

# Import YOLOv5 dependencies
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from collections import deque

# Load model
model_path = 'best.pt'
device = select_device('')
model = DetectMultiBackend(model_path, device=device, dnn=False)
stride, names = model.stride, model.names
#model.warmup(imgsz=(1, 3, 640, 640))  # optional

#image = [x for x in glob.glob("D:/RISET/buoys.v1i.coco/train/*.jpg")]
path = '.\Capture\\testing.mp4'
image = cv.VideoCapture(path)  


FONT = cv.FONT_HERSHEY_SIMPLEX

# Koordinat titik src dapat diubah ke titik koordinat objek yang terdeteksi
# Ukuran perspektif dapat diubah sesuai kebutuhan
def perspective_warp(img,
                     dst_size=(1280, 720),
                     src=np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)]),
                     dst=np.float32([(0, 0), (1, 0), (0, 1), (1, 1)])):
    img_size = np.float32([(img.shape[1], img.shape[0])])
    src = src * img_size
    dst = dst * np.float32(dst_size)
    M = cv.getPerspectiveTransform(src, dst)
    warped = cv.warpPerspective(img, M, dst_size)
    return warped

def get_nearest_objects(obj_coords, ref_point, n=4):
    # obj_coords: list of (x, y) koordinat buoy
    # ref_point: (x_ref, y_ref)
    # n: jumlah objek yang diambil
    if not isinstance(obj_coords, (list, tuple)) or len(obj_coords) == 0:
        return []
    dists = [(pt, np.linalg.norm(np.array(pt) - np.array(ref_point))) for pt in obj_coords]
    dists.sort(key=lambda x: x[1])
    nearest = [pt for pt, _ in dists[:n]]
    return nearest


def inv_perspective_warp(img,
                         dst_size=(1280, 720),
                         src=np.float32([(0, 0), (1, 0), (0, 1), (1, 1)]),
                         dst=np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)])):
    img_size = np.float32([(img.shape[1], img.shape[0])])
    src = src * img_size
    dst = dst * np.float32(dst_size)
    M = cv.getPerspectiveTransform(src, dst)
    warped = cv.warpPerspective(img, M, dst_size)
    return warped


def get_hist(img):
    hist = np.sum(img[img.shape[0] // 2:, :], axis=0)
    return hist



# fungsi dengan tiga parameter : frame gambar, array posisi x untuk objek sisi kiri dan array posisi x untuk objek sisi kanan
def get_curve(img, leftx, rightx):
    leftx = np.array(leftx)
    rightx = np.array(rightx)
    
    # array berisi koordinat y untuk seluruh tinggi gambar
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    # mengambil nilai y terbesar dari koordinat tinggi gambar
    y_eval = np.max(ploty)
    lebar_area = []

    # Interpolasi jika jumlah titik tidak sama dengan ploty
    if len(leftx) < 3 or len(rightx) < 3 :
        return (0, 0, 0)
    
    if len(leftx) != len(ploty) and len(rightx) != len(ploty):
        leftx_interp = np.interp(ploty, np.linspace(0, img.shape[0] - 1, len(leftx)), leftx)
        rightx_interp = np.interp(ploty, np.linspace(0, img.shape[0] - 1, len(rightx)), rightx)
    else:
        leftx_interp = leftx
        rightx_interp = rightx


    for x_kiri, x_kanan in zip(leftx_interp, rightx_interp):
        lebar = abs(x_kanan - x_kiri)
        lebar_area.append(lebar)

    if len(lebar_area) == 0:
        return (0, 0, 0)

    rata2_lebar = int(sum(lebar_area) / len(lebar_area))
    # BAGIAN PENTING -Mengonversi dari pixel ke meter-
    # konversi bergantung dimensi frame (jarak antar objek didapatkan dengan frame bbox YOLO)
    ym_per_pix = 2 / 640 # meter per pixel secara vertikal
    xm_per_pix = rata2_lebar / 640 # meter per pixel secara horizontal
    # Melakukan fitting polinomial orde 2 (kuadratik) pada titik-titik garis kiri dan kanann, namun dalam satuan meter, bukan pixel
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx_interp * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx_interp * xm_per_pix, 2) 
    # Menghitung radius kelengkungan (curvature) untuk garis kiri dan kanan menggunakan rumus matematika kurva polinomial
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    # Mengambil posisi horizontal tengah gambar (posisi kapal pada gambar).
    ship_pos = img.shape[1] // 2
    # Menghitung titik potong sumbu x pada bagian bawah gambar untuk garis kiri dan kanan (menggunakan polinomial hasil fitting)
    l_fit_x_int = left_fit_cr[0] * img.shape[0] ** 2 + left_fit_cr[1] * img.shape[0] + left_fit_cr[2]
    r_fit_x_int = right_fit_cr[0] * img.shape[0] ** 2 + right_fit_cr[1] * img.shape[0] + right_fit_cr[2]
    # Menghitung posisi tengah lajur (lane) berdasarkan rata-rata titik potong kiri dan kanan
    lane_center_position = (r_fit_x_int + l_fit_x_int) / 2
    # BAGIAN PENTING -Menghitung offset posisi kenaraan terhadap tengah lajur-
    # Menghitung seberapa jauh posisi kapal dari tengah lajur, dikonversi ke meter (dan dibagi 10, kemungkinan untuk normalisasi).
    center = (ship_pos - lane_center_position) * xm_per_pix 
    # Mengembalikan tuple yang berisi radius kelengkungan kiri, kanan, dan posisi relatif kapal terhadap tengah lajur.
    return (left_curverad, right_curverad, center)



def calc_servo_pos(center_value):
    main_offset = abs(center_value)
    if main_offset < 10:
        return 'ALMOST CENTER', 'CENTER', 0
    elif 10 < main_offset < 100:
        if center_value > 0:
            return 'TURN SLIGHTLY TO LEFT', 'LEFT', -30
        else:
            return 'TURN SLIGHTLY TO RIGHT', 'RIGHT', 30
    elif main_offset > 100:
        if center_value > 0:
            return 'TURN HARDLY TO LEFT', 'LEFT', -75
        else:
            return 'TURN HARDLY TO RIGHT', 'RIGHT', 75
    else:
        return 'NONE', 'NONE', 'NONE'



def detect_object(frame) :
    img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).to(device)
    img_tensor = img_tensor.permute(2, 0, 1).float() / 255.0  # CHW, normalize
    img_tensor = img_tensor.unsqueeze(0)  # add batch dim
    red_centers = deque(maxlen=6)
    green_centers = deque(maxlen=6)

    with torch.no_grad():
        # Lower confidence threshold to 0.25
        pred = model(img_tensor, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres=0.85, iou_thres=0.35)

        red_centers.clear()
        green_centers.clear()
      
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

                    cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv.putText(frame, f"{class_name}", (x1, y1 - 10),
                                FONT, 0.6, color, 2)

        # Urutkan dari bawah ke atas (Y besar ke kecil)
        red_centers = sorted(red_centers, key=lambda pt: -pt[1])
        green_centers = sorted(green_centers, key=lambda pt: -pt[1])

    # Debug info
    print(f"[DEBUG] red_buoy: {len(red_centers)}, green_buoy: {len(green_centers)}")
    return  red_centers, green_centers

def vid_pipeline(img):
    # menyimpan koordinat floating ball merah dan hijau 
    Red_centers = []
    Green_centers = []
    # memproses gambar dengan fungsi detect_object, untuk mendeteksi tepi garis lajur  
    Red_centers, Green_centers  = detect_object(img) 
    all_buoys = Red_centers + Green_centers
    h, w = img.shape[:2]
    ref_point = (w//2, h)  # misal, titik bawah tengah gambar
    nearest_4 = get_nearest_objects(all_buoys, ref_point, n=4)

    # melakukan transformasi perspektif pada hasil thresholding sehingga tampak seperti pandangan atas unuk memudahkan deteksi lajur
    if len(nearest_4) == 4:
        src = np.float32(nearest_4)
        # dst tetap seperti biasa
        warped = perspective_warp(img, src=src)
    else:
        # fallback ke default src
        warped = perspective_warp(img)

    # mendeteksi posisi lajur kiri dan kanan menggunakan metode sliding window pada gambar hasil warp. menghasilkan koordinat lajur dari gambar hasil deteksi. 
    # menghitung radius kelengkunagan lajur dan posisi kendaraan terhadap tengah lajur
    red_x = [x for x, y in Red_centers]
    green_x = [x for x, y in Green_centers]
    # Contoh penggunaan nearest_pair
    # red_pairs, green_pairs = nearest_pair(Red_centers, Green_centers)
    curverad = get_curve(img, red_x, green_x)

    # mengambil rata-rata radius kelengkungan dari lajur kiri dan kanan 
    lane_curve = np.mean([curverad[0], curverad[1]])
    
    # menghitung posisi rata-rata ujung lajur kiri dan kanan
    left_most_end = np.mean(red_x)
    right_most_end = np.mean(green_x)
    
    # menentukan jenis font untuk teks yang akan ditampilkan pada gambar
    
    # menampilkan informasi kelengkungan lajur dan posisi kendaraan
    cv.putText(img, 'Lane Curvature: {:.0f} m'.format(lane_curve), (10, 20), FONT, 0.4, (0, 255, 255), 1)
    cv.putText(img, 'Vehicle offset: {:.4f} m'.format(curverad[2]), (10, 40), FONT, 0.4, (0, 255, 255), 1)
    
    # menggambar garis vertikal ditangah bawah gambar, biasanya sebagai referensi posisi kendaraan
    cv.line(img, (640, 535), (640, 595), (0, 0, 255), 2)
    
    # mengembalikan gambar hasil akhir dan posisi rata - rata ujung lajur kiri dan kanan.
    return img, left_most_end, right_most_end



while True:
    start_time = time.time()
    ret, frame = image.read()
    if not ret :
        break

    img = cv.resize(frame, (640, 640), interpolation=cv.INTER_AREA)
    kp, left_most_end, right_most_end = vid_pipeline(img)
    # REFACTOR THIS
    if np.isnan(left_most_end) or np.isnan(right_most_end) or left_most_end == 0 and right_most_end == 0:
        lane_center_pos = w // 2  # fallback ke tengah gambar
    else:
        lane_center_pos = int((right_most_end - left_most_end) / 2) + int(left_most_end)
    # Ukuran dinamis
    h, w = kp.shape[:2]
    # Teks dan slider proporsional
    cv.putText(kp, f'Processing Time: {round((time.time() - start_time), 4)} Seconds', (10,60),
        FONT, 0.4, (0, 255, 255), 1)
    cv.rectangle(kp, (int(w*0.1), int(h*0.84)), (int(w*0.8), int(h*0.93)), (0, 0, 255), 5)
    cv.putText(kp, 'Steering Control', (int(w*0.4), int(h*0.81)), FONT, 1, (0, 255, 0), 2)
    cv.putText(kp, '.', (int(lane_center_pos), int(h*0.89)), FONT , 2, (0, 255, 0), 25)
    offset = int((w//2) - lane_center_pos)
    steering_commands, side, PWM = calc_servo_pos(offset)
    cv.putText(kp, f"Remarks: {steering_commands} ", (10,80), FONT , 0.4, (0, 255, 255), 1)
    print(steering_commands)
    cv.putText(kp, f"Servo Commands: SIDE={side} PWM(DC)={PWM}% ", (10,100), FONT , 0.4,
        (0, 255, 255), 1)
    cv.putText(kp,f"Image Size : {kp.shape[0]} * {kp.shape[1]}", (10,120),FONT, 0.4, (0,255,255), 1 )
    cv.imshow('Lane Detection', kp)
  

    keyCode = cv.waitKey(1)
    if keyCode == ord('q'):
        break

cv.destroyAllWindows()