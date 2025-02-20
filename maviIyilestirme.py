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

# Geliştirilmiş mavi renk aralıkları - daha geniş spektrum
lower_blue1 = np.array([90, 100, 100])   # Daha düşük doygunluk ve parlaklık
upper_blue1 = np.array([130, 255, 255])  # Daha geniş mavi spektrum

# Açık mavi için ek aralık
lower_blue2 = np.array([85, 50, 150])    # Açık maviler için
upper_blue2 = np.array([95, 255, 255])   # Açık maviler için

# Kamera optimizasyonları
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)    # Çözünürlük artırıldı
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)   # Çözünürlük artırıldı
cap.set(cv2.CAP_PROP_FPS, 30)             # FPS optimize edildi
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75) # Otomatik pozlama ayarı
cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)     # Parlaklık ayarı

if not cap.isOpened():
    print("Kamera başlatılamadı!")
    exit()

# Gelişmiş morfolojik filtreleme
kernel_noise = np.ones((3, 3), np.uint8)   # Gürültü temizleme için küçük kernel
kernel_close = np.ones((7, 7), np.uint8)   # Boşluk kapatma için büyük kernel

def enhance_blue_detection(frame):
    """Mavi renk tespitini iyileştiren fonksiyon"""
    # Gaussian bulanıklaştırma uygula
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    
    # HSV dönüşümü
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # İki mavi aralığı için maske oluştur
    mask1 = cv2.inRange(hsv, lower_blue1, upper_blue1)
    mask2 = cv2.inRange(hsv, lower_blue2, upper_blue2)
    
    # Maskeleri birleştir
    mask_blue = cv2.bitwise_or(mask1, mask2)
    
    # Gürültü temizleme
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel_noise)
    
    # Boşlukları kapatma
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel_close)
    
    # Kenar yumuşatma
    mask_blue = cv2.GaussianBlur(mask_blue, (3, 3), 0)
    
    return mask_blue

# Frame sayacı
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Kamera bağlantısı kesildi, tekrar bağlanılıyor...")
        cap = cv2.VideoCapture(0)
        continue

    frame_count += 1
    if frame_count % 2 != 0:  # Her 2 frame'den birini işle
        continue

    # Görüntü ön işleme
    frame = cv2.bilateralFilter(frame, 9, 75, 75)  # Kenarları koruyarak yumuşatma
    
    # YOLO tespiti
    results = model(frame, stream=True)
    detections = []

    # Geliştirilmiş mavi tespit
    mask_blue = enhance_blue_detection(frame)
    
    # Mavi maske ekranı göster
    cv2.imshow("Blue Mask", mask_blue)

    # YOLO tespitlerini işleme
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if x1 < 0 or y1 < 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0]:
                continue

            # Geliştirilmiş mavi piksel kontrolü
            roi = mask_blue[y1:y2, x1:x2]
            if roi.size > 0:
                blue_ratio = cv2.countNonZero(roi) / roi.size
                if blue_ratio > 0.2:  # Eşik değeri düşürüldü
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    detections.append([[x1, y1, x2, y2], conf, cls])

                    # Mavi algılandığında gelişmiş görselleştirme
                    # Tespit güvenirliğine göre renk tonu
                    confidence_color = (255, int(255 * (1 - blue_ratio)), 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), confidence_color, 2)
                    cv2.putText(frame, f"Blue {blue_ratio:.2f}", (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, confidence_color, 2)

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

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
        cv2.putText(frame, str(track.track_id), (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.imshow("Tracking", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()