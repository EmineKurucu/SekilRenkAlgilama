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
# Mavi renk aralıkları
lower_blue1 = np.array([90, 100, 100])
upper_blue1 = np.array([130, 255, 255])
lower_blue2 = np.array([85, 50, 150])
upper_blue2 = np.array([95, 255, 255])

# Yeşil renk aralıkları
lower_green1 = np.array([40, 100, 100])
upper_green1 = np.array([80, 255, 255])
lower_green2 = np.array([35, 50, 150])
upper_green2 = np.array([85, 255, 255])

# Kırmızı renk aralıkları
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

# Kamera optimizasyonları
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

# Morfolojik işlem kernelleri
kernel_noise = np.ones((3, 3), np.uint8)
kernel_close = np.ones((7, 7), np.uint8)

def enhance_color_detection(frame, lower1, upper1, lower2, upper2):
    """Renk tespitini iyileştiren genel fonksiyon"""
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # İki aralık için maske oluştur
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    
    # Maskeleri birleştir
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Gürültü temizleme ve boşluk kapatma
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

    # Görüntü ön işleme
    frame = cv2.bilateralFilter(frame, 9, 75, 75)
    
    # YOLO tespiti
    results = model(frame, stream=True)
    detections = []

    # Her renk için maske oluştur
    mask_blue = enhance_color_detection(frame, lower_blue1, upper_blue1, lower_blue2, upper_blue2)
    mask_green = enhance_color_detection(frame, lower_green1, upper_green1, lower_green2, upper_green2)
    mask_red = enhance_color_detection(frame, lower_red1, upper_red1, lower_red2, upper_red2)
    
    # Maskeleri göster
    cv2.imshow("Blue Mask", mask_blue)
    cv2.imshow("Green Mask", mask_green)
    cv2.imshow("Red Mask", mask_red)

    # YOLO tespitlerini işleme
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if x1 < 0 or y1 < 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0]:
                continue

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            # Her renk için oran hesapla
            blue_ratio = cv2.countNonZero(mask_blue[y1:y2, x1:x2]) / roi.size
            green_ratio = cv2.countNonZero(mask_green[y1:y2, x1:x2]) / roi.size
            red_ratio = cv2.countNonZero(mask_red[y1:y2, x1:x2]) / roi.size

            # En yüksek orana sahip rengi belirle
            ratios = {
                "Blue": (blue_ratio, (255, 0, 0)),
                "Green": (green_ratio, (0, 255, 0)),
                "Red": (red_ratio, (0, 0, 255))
            }
            
            max_color = max(ratios.items(), key=lambda x: x[1][0])
            if max_color[1][0] > 0.2:  # Renk oranı eşiği
                color_name = max_color[0]
                color_rgb = max_color[1][1]
                ratio = max_color[1][0]

                conf = float(box.conf[0])
                cls = int(box.cls[0])
                detections.append([[x1, y1, x2, y2], conf, cls])

                # Tespit edilen renk için görselleştirme
                confidence_factor = int(255 * ratio)
                display_color = tuple(int(c * confidence_factor/255) for c in color_rgb)
                cv2.rectangle(frame, (x1, y1), (x2, y2), display_color, 2)
                cv2.putText(frame, f"{color_name} {ratio:.2f}", (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, display_color, 2)

    # DeepSORT tracking
    try:
        tracks = tracker.update_tracks(detections, frame=frame)
    except Exception as e:
        print(f"Tracker hatası: {e}")
        tracks = []

    # Tracking görselleştirme
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

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()