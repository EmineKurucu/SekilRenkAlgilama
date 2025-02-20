import cv2
import numpy as np

# Kamera ayarları
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def detect_shape(contour):
    """Konturun şeklini tespit eder"""
    # Şekli yaklaşık bir poligona dönüştür
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    
    # Köşe sayısına göre şekli belirle
    corners = len(approx)
    
    # Dairesellik hesapla
    area = cv2.contourArea(contour)
    if area == 0:
        return "unknown"
        
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    if circularity > 0.85:  # Dairesellik eşiği
        return "circle"
    elif corners == 3:
        return "triangle"
    elif corners == 4:
        # Kare/dikdörtgen ayrımı için en-boy oranı kontrolü
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w)/h
        if 0.95 <= aspect_ratio <= 1.05:  # Kare için oran kontrolü
            return "square"
        else:
            return "rectangle"
    else:
        return "unknown"

def create_shape_mask(frame, shape_name):
    """Belirli bir şekil için maske oluşturur"""
    # Gri tonlamaya çevir
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Gürültü azaltma
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Kenar tespiti
    edges = cv2.Canny(blurred, 50, 150)
    
    # Morfolojik işlemler
    kernel = np.ones((5,5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    
    # Konturları bul
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Boş maske oluştur
    mask = np.zeros_like(gray)
    
    # Her kontur için
    for contour in contours:
        # Küçük konturları filtrele
        if cv2.contourArea(contour) < 500:  # Alan eşiği
            continue
            
        # Şekli tespit et
        detected_shape = detect_shape(contour)
        
        # İstenen şekil ise maskeye ekle
        if detected_shape == shape_name:
            cv2.drawContours(mask, [contour], -1, (255), -1)
    
    return mask

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera bağlantısı kesildi!")
        break
    
    # Her şekil için maske oluştur
    circle_mask = create_shape_mask(frame, "circle")
    triangle_mask = create_shape_mask(frame, "triangle")
    square_mask = create_shape_mask(frame, "square")
    
    # Orijinal görüntü üzerinde şekilleri işaretle
    display_frame = frame.copy()
    
    # Her şekil için konturları bul ve göster
    shapes = {
        "circle": (circle_mask, (0, 0, 255)),    # Kırmızı
        "triangle": (triangle_mask, (0, 255, 0)), # Yeşil
        "square": (square_mask, (255, 0, 0))     # Mavi
    }
    
    for shape_name, (mask, color) in shapes.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Kontur çiz
            cv2.drawContours(display_frame, [contour], -1, color, 2)
            
            # Şekil adını yaz
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(display_frame, shape_name, (cx-20, cy),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Görüntüleri göster
    cv2.imshow("Original with Shapes", display_frame)
    cv2.imshow("Circle Mask", circle_mask)
    cv2.imshow("Triangle Mask", triangle_mask)
    cv2.imshow("Square Mask", square_mask)
    
    # Çıkış kontrolü
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()