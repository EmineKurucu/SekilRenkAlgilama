# Testte iyi sonuç veriyor
# YEŞİL MVİ KIRMIZI RENKLERİMİN ALGILANDIĞI YERDE OPENCV İLE ŞEKİL TESPİTİ YAPIYOR
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

# Renk aralıkları tanımlamaları aynı kalıyor
lower_blue1 = np.array([90, 100, 100])
upper_blue1 = np.array([130, 255, 255])
lower_blue2 = np.array([85, 50, 150])
upper_blue2 = np.array([95, 255, 255])

lower_green1 = np.array([40, 100, 100])
upper_green1 = np.array([80, 255, 255])
lower_green2 = np.array([35, 50, 150])
upper_green2 = np.array([85, 255, 255])

lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

# Kamera optimizasyonları aynı
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)

if not cap.isOpened():
    print("Kamera başlatılamadı!")
    exit()

kernel_noise = np.ones((3, 3), np.uint8)
kernel_close = np.ones((7, 7), np.uint8)

def detect_shape(contour):
    """Konturun şeklini tespit eden fonksiyon"""
    shape = "Belirsiz"
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    
    # Kontur alanı çok küçükse yoksay
    if cv2.contourArea(contour) < 100:
        return None
    
    if len(approx) == 3:
        shape = "Ucgen"
    elif len(approx) == 4:
        # Kare mi dikdörtgen mi kontrol et
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w)/h
        if 0.95 <= aspect_ratio <= 1.05:
            shape = "Kare"
        else:
            shape = "Dikdortgen"
    else:
        # Daire tespiti için dairesellik oranı
        area = cv2.contourArea(contour)
        circularity = 4 * np.pi * area / (peri * peri)
        if circularity > 0.8:
            shape = "Daire"
            
    return shape

def enhance_color_detection(frame, lower1, upper1, lower2, upper2):
    """Renk tespitini iyileştiren genel fonksiyon"""
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_noise)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    
    return mask

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Kamera bağlantısı kesildi, tekrar bağlanılıyor...")
        cap = cv2.VideoCapture(0)
        continue

    frame_count += 1
    if frame_count % 2 != 0:
        continue

    frame = cv2.bilateralFilter(frame, 9, 75, 75)
    
    results = model(frame, stream=True)
    detections = []

    mask_blue = enhance_color_detection(frame, lower_blue1, upper_blue1, lower_blue2, upper_blue2)
    mask_green = enhance_color_detection(frame, lower_green1, upper_green1, lower_green2, upper_green2)
    mask_red = enhance_color_detection(frame, lower_red1, upper_red1, lower_red2, upper_red2)
    
    # Her renk için kontur ve şekil tespiti
    color_masks = {
        "Mavi": (mask_blue, (255, 0, 0)),
        "Yesil": (mask_green, (0, 255, 0)),
        "Kirmizi": (mask_red, (0, 0, 255))
    }
    
    for color_name, (mask, color) in color_masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            shape = detect_shape(contour)
            if shape:
                # Konturun sınırlayıcı dikdörtgenini al
                x, y, w, h = cv2.boundingRect(contour)
                
                # Şekil ve renk bilgisini çiz
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label = f"{color_name} {shape}"
                cv2.putText(frame, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # YOLO tespitleri için detection listesine ekle
                detections.append([[x, y, x + w, y + h], 0.9, 0])  # confidence 0.9 ve class 0 varsayılan

    # YOLO ve tracking işlemleri devam ediyor
    try:
        tracks = tracker.update_tracks(detections, frame=frame)
    except Exception as e:
        print(f"Tracker hatası: {e}")
        tracks = []

    for track in tracks:
        if not track.is_confirmed():
            continue

        bbox = track.to_tlbr()
        x1, y1, x2, y2 = map(int, bbox)

        if x1 < 0 or y1 < 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0]:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv2.putText(frame, str(track.track_id), (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Tracking", frame)
    cv2.imshow("Blue Mask", mask_blue)
    cv2.imshow("Green Mask", mask_green)
    cv2.imshow("Red Mask", mask_red)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()