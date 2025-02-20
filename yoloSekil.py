import cv2
import numpy as np
from ultralytics import YOLO
import torch

# Model yükleme ve optimizasyonları
model = YOLO("yolov8n.pt")
model.conf = 0.25  # Tespit eşiği
model.iou = 0.45   # IOU eşiği

# Kamera ayarları
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

def create_mask(frame, bbox):
    """Tespit edilen nesne için maske oluşturur"""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    x1, y1, x2, y2 = map(int, bbox)
    mask[y1:y2, x1:x2] = 255
    return mask

def detect_shape(roi):
    """ROI içindeki şekli analiz eder"""
    # Gri tonlamaya çevir
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Eşikleme
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Konturları bul
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return "unknown"
        
    # En büyük konturu al
    contour = max(contours, key=cv2.contourArea)
    
    # Şekli yaklaşık bir poligona dönüştür
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    
    # Köşe sayısı
    corners = len(approx)
    
    # Dairesellik hesapla
    area = cv2.contourArea(contour)
    if area == 0:
        return "unknown"
        
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    if circularity > 0.85:
        return "circle"
    elif corners == 3:
        return "triangle"
    elif corners == 4:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w)/h
        if 0.95 <= aspect_ratio <= 1.05:
            return "square"
        else:
            return "rectangle"
    else:
        return "unknown"

# Şekil maskeleri için sözlük
shape_masks = {
    "circle": None,
    "triangle": None,
    "square": None
}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera bağlantısı kesildi!")
        break
    
    # Her frame için maskeleri sıfırla
    for shape in shape_masks:
        shape_masks[shape] = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    # YOLO tespiti
    results = model(frame, stream=True)
    
    # Tespit edilen nesneleri işle
    for result in results:
        boxes = result.boxes
        
        for box in boxes:
            # Sınırlayıcı kutu koordinatları
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # ROI çıkar
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            
            # Şekli tespit et
            shape = detect_shape(roi)
            
            if shape in shape_masks:
                # Şekil maskesini güncelle
                mask = create_mask(frame, [x1, y1, x2, y2])
                shape_masks[shape] = cv2.bitwise_or(shape_masks[shape], mask)
                
                # Şekli çiz ve etiketle
                color = {
                    "circle": (0, 0, 255),    # Kırmızı
                    "triangle": (0, 255, 0),  # Yeşil
                    "square": (255, 0, 0)     # Mavi
                }[shape]
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, shape, (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Görüntüleri göster
    cv2.imshow("YOLOv8 Shape Detection", frame)
    
    # Her şekil maskesini göster
    for shape, mask in shape_masks.items():
        cv2.imshow(f"{shape.capitalize()} Mask", mask)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()