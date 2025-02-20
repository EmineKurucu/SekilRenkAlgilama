import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Model optimizasyonları
model = YOLO("yolov8n.pt")
model.conf = 0.3
model.iou = 0.4

# DeepSORT optimizasyonları
tracker = DeepSort(
    max_age=3,
    n_init=1,
    max_iou_distance=0.3,
    max_cosine_distance=0.25,
    nn_budget=None,
)

# Renk aralıkları tanımlamaları
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])

lower_green = np.array([40, 50, 50])
upper_green = np.array([80, 255, 255])

lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 50, 50])
upper_red2 = np.array([180, 255, 255])

# Kamera optimizasyonları
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Kamera başlatılamadı!")
    exit()

kernel_noise = np.ones((3, 3), np.uint8)
kernel_close = np.ones((7, 7), np.uint8)

def detect_shape(contour):
    """Konturun şeklini tespit eden fonksiyon"""
    shape = "Belirsiz"
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    
    if len(approx) == 3:
        shape = "Ucgen"
    elif len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w)/h
        shape = "Kare" if 0.95 <= aspect_ratio <= 1.05 else "Dikdortgen"
    else:
        area = cv2.contourArea(contour)
        circularity = 4 * np.pi * area / (peri * peri)
        if circularity > 0.8:
            shape = "Daire"
    
    return shape

def enhance_color_detection(frame, lower, upper):
    """Renk tespitini iyileştiren genel fonksiyon"""
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_noise)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    return mask

# Konsol temizleme ve hedef bilgisi yazdırma fonksiyonu
def print_target_info(color, shape, status):
    """Hedef bilgisini konsola yazdırır"""
    if status == "searching":
        print("\nHedef aranıyor...", end='\r')
    elif status == "found":
        print(f"\nHEDEF TESPİT EDİLDİ!")
        print(f"------------------")
        print(f"Renk: {color}")
        print(f"Şekil: {shape}")
        print(f"------------------")
    elif status == "tracking":
        print(f"Takip ediliyor: {color} {shape}", end='\r')
    elif status == "reset":
        print("\nHedef sıfırlandı. Yeni hedef aranıyor...")

target_color = None
target_shape = None
target_found = False
frame_count = 0

print("\nProgram başlatıldı. Hedef aranıyor...")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("\nKamera bağlantısı kesildi, tekrar bağlanılıyor...")
        cap = cv2.VideoCapture(0)
        continue

    frame_count += 1
    if frame_count % 2 != 0:
        continue

    frame = cv2.bilateralFilter(frame, 9, 75, 75)

    if not target_found:
        print_target_info(None, None, "searching")

    mask_blue = enhance_color_detection(frame, lower_blue, upper_blue)
    mask_green = enhance_color_detection(frame, lower_green, upper_green)
    mask_red1 = enhance_color_detection(frame, lower_red1, upper_red1)
    mask_red2 = enhance_color_detection(frame, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    color_masks = {
        "Mavi": (mask_blue, (255, 0, 0)),
        "Yesil": (mask_green, (0, 255, 0)),
        "Kirmizi": (mask_red, (0, 0, 255))
    }

    for color_name, (mask, color) in color_masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 500:  # Minimum alan kontrolü
                continue
                
            shape = detect_shape(cnt)
            x, y, w, h = cv2.boundingRect(cnt)

            if target_found:
                if color_name == target_color and shape == target_shape:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    print_target_info(target_color, target_shape, "tracking")
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                if target_color is None and target_shape is None:
                    target_color = color_name
                    target_shape = shape
                    target_found = True
                    print_target_info(target_color, target_shape, "found")

    cv2.imshow("Tracking", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        print("\nProgram sonlandırılıyor...")
        break
    elif key == ord('r'):
        target_color = None
        target_shape = None
        target_found = False
        print_target_info(None, None, "reset")

cap.release()
cv2.destroyAllWindows()