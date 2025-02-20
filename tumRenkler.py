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
# Kırmızı renk aralıkları
lower_red1 = np.array([0, 150, 150])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 150, 150])
upper_red2 = np.array([180, 255, 255])

# Mavi renk aralığı
lower_blue = np.array([100, 150, 150])
upper_blue = np.array([130, 255, 255])

# Yeşil renk aralığı
lower_green = np.array([40, 150, 150])
upper_green = np.array([80, 255, 255])

# Kamera optimizasyonları
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Kamera başlatılamadı!")
    exit()

# Morfolojik işlemler için kernel
kernel = np.ones((5, 5), np.uint8)

# Frame sayacı
frame_count = 0

def create_color_mask(hsv_frame, lower_bounds, upper_bounds):
    """Belirli renk aralığı için maske oluşturur"""
    if isinstance(lower_bounds, list):
        # Kırmızı gibi çift aralıklı renkler için
        mask = cv2.inRange(hsv_frame, lower_bounds[0], upper_bounds[0])
        mask2 = cv2.inRange(hsv_frame, lower_bounds[1], upper_bounds[1])
        return mask | mask2
    else:
        # Tek aralıklı renkler için
        return cv2.inRange(hsv_frame, lower_bounds, upper_bounds)

def process_color_detection(frame, mask, color_name):
    """Renk tespiti ve görselleştirme işlemleri"""
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Kamera bağlantısı kesildi, tekrar bağlanılıyor...")
        cap = cv2.VideoCapture(0)
        continue
    
    frame_count += 1
    if frame_count % 2 != 0:
        continue

    # YOLO tespiti
    results = model(frame, stream=True)
    detections = []

    # HSV dönüşümü
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Renk maskeleri oluşturma
    mask_red = create_color_mask(hsv, [lower_red1, lower_red2], [upper_red1, upper_red2])
    mask_blue = create_color_mask(hsv, lower_blue, upper_blue)
    mask_green = create_color_mask(hsv, lower_green, upper_green)

    # Renk maskelerini işleme
    mask_red = process_color_detection(frame, mask_red, "Red")
    mask_blue = process_color_detection(frame, mask_blue, "Blue")
    mask_green = process_color_detection(frame, mask_green, "Green")

    # Maskeleri göster
    cv2.imshow("Red Mask", mask_red)
    cv2.imshow("Blue Mask", mask_blue)
    cv2.imshow("Green Mask", mask_green)

    # YOLO tespitlerini işleme
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            if x1 < 0 or y1 < 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0]:
                continue

            # Her renk için ROI kontrolü
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Her renk için piksel oranı kontrolü
            red_ratio = cv2.countNonZero(mask_red[y1:y2, x1:x2]) / roi.size
            blue_ratio = cv2.countNonZero(mask_blue[y1:y2, x1:x2]) / roi.size
            green_ratio = cv2.countNonZero(mask_green[y1:y2, x1:x2]) / roi.size
            
            color_detected = None
            box_color = None
            
            # En baskın rengi belirleme
            if max(red_ratio, blue_ratio, green_ratio) > 0.25:
                if red_ratio == max(red_ratio, blue_ratio, green_ratio):
                    color_detected = "Red"
                    box_color = (0, 0, 255)
                elif blue_ratio == max(red_ratio, blue_ratio, green_ratio):
                    color_detected = "Blue"
                    box_color = (255, 0, 0)
                else:
                    color_detected = "Green"
                    box_color = (0, 255, 0)
                
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                detections.append([[x1, y1, x2, y2], conf, cls])

                # Tespit edilen renk için çerçeve ve etiket
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, f"{color_detected} detected", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

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

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(frame, str(track.track_id), (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow("Tracking", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()