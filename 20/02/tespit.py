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

# Renk aralıkları iyileştirildi
lower_blue = np.array([95, 50, 50])
upper_blue = np.array([125, 255, 255])

lower_green = np.array([45, 50, 50])
upper_green = np.array([75, 255, 255])

lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 50, 50])
upper_red2 = np.array([180, 255, 255])

# Kamera optimizasyonları
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Otomatik odaklamayı kapat
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Otomatik pozlamayı kapat

if not cap.isOpened():
    print("Kamera başlatılamadı!")
    exit()

# Gürültü giderme ve morfolojik işlemler için kerneller
kernel_noise = np.ones((3, 3), np.uint8)
kernel_close = np.ones((7, 7), np.uint8)

class ShapeDetector:
    @staticmethod
    def get_shape_score(contour):
        """Şekil tespiti için güven skoru hesaplar"""
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        area = cv2.contourArea(contour)
        
        if area < 100:  # Çok küçük konturları ele
            return "Belirsiz", 0
            
        len_approx = len(approx)
        scores = {
            "Ucgen": 0,
            "Kare": 0,
            "Daire": 0
        }
        
        # Üçgen skoru
        if len_approx == 3:
            scores["Ucgen"] = 0.9
        
        # Kare skoru
        elif len_approx == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w)/h
            if 0.95 <= aspect_ratio <= 1.05:
                scores["Kare"] = 0.9
        
        # Daire skoru
        circularity = 4 * np.pi * area / (peri * peri)
        if 0.85 <= circularity <= 1.15:
            scores["Daire"] = 0.9
            
        best_shape = max(scores.items(), key=lambda x: x[1])
        return best_shape if best_shape[1] > 0.5 else ("Belirsiz", 0)

def enhance_color_detection(frame, lower, upper):
    """Geliştirilmiş renk tespiti"""
    # Gürültü azaltma
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    median = cv2.medianBlur(blurred, 5)
    
    # HSV dönüşümü ve maske oluşturma
    hsv = cv2.cvtColor(median, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    
    # Morfolojik işlemler
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_noise)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask = cv2.dilate(mask, kernel_noise, iterations=1)
    
    return mask

def draw_target_info(frame, target_color, target_shape, target_found):
    """Hedef bilgisini ekrana yazdırır"""
    info_color = (0, 255, 255)  # Sarı
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    if target_found:
        info_text = f"HEDEF =>  {target_color}, {target_shape}"
        cv2.putText(frame, info_text, (20, 30), font, 0.7, info_color, 2)
        cv2.putText(frame, "Durum: Takip Ediliyor", (20, 60), font, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Hedef Belirleniyor...", (20, 30), font, 0.7, info_color, 2)

# Ana değişkenler
target_color = None
target_shape = None
target_found = False
shape_detector = ShapeDetector()
frame_count = 0

# Hedef log dosyası
with open('hedef_log.txt', 'w') as f:
    f.write("Hedef Takip Logu\n")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Kamera bağlantısı kesildi, tekrar bağlanılıyor...")
        cap = cv2.VideoCapture(0)
        continue

    frame_count += 1
    if frame_count % 2 != 0:  # Her 2 karede bir işlem yap
        continue

    # Görüntü ön işleme
    frame = cv2.bilateralFilter(frame, 9, 75, 75)

    # Renk maskeleri
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
            area = cv2.contourArea(cnt)
            if area < 500:  # Minimum alan kontrolü
                continue
                
            shape, confidence = shape_detector.get_shape_score(cnt)
            if shape == "Belirsiz":
                continue
                
            x, y, w, h = cv2.boundingRect(cnt)
            
            if target_found:
                if color_name == target_color and shape == target_shape:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f"Hedef: {color_name} {shape}", (x, y - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{color_name} {shape}", (x, y - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                if target_color is None and target_shape is None:
                    target_color = color_name
                    target_shape = shape
                    target_found = True
                    # Hedefi log dosyasına yaz
                    with open('hedef_log.txt', 'a') as f:
                        f.write(f"\nYeni Hedef Belirlendi:\nRenk: {target_color}\nŞekil: {target_shape}\n")
                    print(f"\nHEDEF BELİRLENDİ:\nRenk: {target_color}\nŞekil: {target_shape}")

    # Hedef bilgisini ekrana yazdır
    draw_target_info(frame, target_color, target_shape, target_found)

    # Görüntüleri göster
    cv2.imshow("Tracking", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
    elif key == ord('r'):
        target_color = None
        target_shape = None
        target_found = False
        print("\nHedef sıfırlandı. Yeni hedef bekleniyor...")
        with open('hedef_log.txt', 'a') as f:
            f.write("\nHedef Sıfırlandı\n")

cap.release()
cv2.destroyAllWindows()