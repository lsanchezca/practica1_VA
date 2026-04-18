import cv2
import numpy as np
import os
import argparse

def aplicar_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

class PanelDetector:
    def __init__(self):
        # Ajustar parámetros de MSER para evitar ruido
        self.mser = cv2.MSER_create(
            delta=5,
            min_area=60,
            max_area=50000,)


    def calcular_score(self, roi):
        # 1. Redimensionar ROI a 40x80
        roi_resized = cv2.resize(roi, (80, 40))

        # 2. Convertir a HSV
        hsv = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2HSV)

        # 3. Rango azul
        lower_blue = np.array([100, 100, 50])
        upper_blue = np.array([130, 255, 255])

        # 4. Máscara azul del ROI redimensionado
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        mask = (mask > 0).astype(np.float32)

        # 5. Máscara ideal (todo azul)
        ideal_mask = np.ones((40, 80), dtype=np.float32)

        # 6. Score = proporción de píxeles azules
        score = np.sum(mask * ideal_mask) / (40 * 80)

        return float(score)
    
    def calcular_thresold(self, boxA, boxB):
        # Calcular el área de intersección
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # Calcular el área de cada caja
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # Calcular el IoU
        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou
            
    def merge_connected_boxes(self, boxes, iou_threshold=0.05, dist_threshold=15):
        if not boxes:
            return []

        groups = []

        for box in boxes:
            added = False
            for g in groups:
                for b in g:
                    iou = self.calcular_thresold(box, b)

                    # distancia entre cajas
                    dx = max(0, max(b[0], box[0]) - min(b[2], box[2]))
                    dy = max(0, max(b[1], box[1]) - min(b[3], box[3]))
                    dist = (dx*dx + dy*dy)**0.5

                    if iou > iou_threshold or dist < dist_threshold:
                        g.append(box)
                        added = True
                        break
                if added:
                    break

            if not added:
                groups.append([box])

        # fusionar cada grupo en una sola caja
        merged = []
        for g in groups:
            x1 = min(b[0] for b in g)
            y1 = min(b[1] for b in g)
            x2 = max(b[2] for b in g)
            y2 = max(b[3] for b in g)
            score = max(b[4] for b in g)
            merged.append([x1, y1, x2, y2, score])

        return merged


    def detectar(self, image):
        # Convertir a gris para MSER
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # histograma del gris (imprimir luego memoria)
        hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
        # mejorar el contraste
        contrast_img = cv2.equalizeHist(gray)
        #filtro gamma
        contrast_img = aplicar_gamma(contrast_img, gamma=1.5)
        # histograma del contraste mejorado (imprimor luego memoria)
        hist_contrast = cv2.calcHist([contrast_img], [0], None, [256], [0, 256])

        regions, _ = self.mser.detectRegions(contrast_img)
        
        detecciones = []
        for p in regions:
            x, y, w, h = cv2.boundingRect(p)
            
            # Filtrado por relación de aspecto
            aspect_ratio = w / float(h)
            if 0.5 < aspect_ratio < 4.0:
                # Agrandar rectángulo para incluir borde blanco
                pad_w = int(0.25 * w)
                pad_h = int(0.25 * h)

                x_exp = max(0, x - pad_w)
                y_exp = max(0, y - pad_h)
                x2_exp = min(image.shape[1], x + w + pad_w)
                y2_exp = min(image.shape[0], y + h + pad_h)

                roi = image[y_exp:y2_exp, x_exp:x2_exp]
                score = self.calcular_score(roi)

                if score > 0.4:
                    detecciones.append([x_exp, y_exp, x2_exp, y2_exp, score])

        
        return self.merge_connected_boxes(detecciones, iou_threshold=0.5)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Trains and executes a given detector over a set of testing images')
    parser.add_argument(
        '--detector', type=str, nargs="?", default="", help='Detector string name')
    parser.add_argument(
        '--train_path', default="", help='Select the training data dir')
    parser.add_argument(
        '--test_path', default="", help='Select the testing data dir')

    args = parser.parse_args()

    # Load training data
    if not os.path.exists("resultado_imgs"):
        os.makedirs("resultado_imgs")

    # Create the detector
    detector = PanelDetector()

    # Load testing data
    if args.test_path:
        # Obtenemos lista de imágenes .png del directorio de test
        test_images = [f for f in os.listdir(args.test_path) if f.endswith('.png')]
        
        # Abrimos el fichero de resultados para escritura
        with open("resultado.txt", "w") as f_res:
            
            # 3. Evaluate detections (Procesar cada imagen)
            for img_name in test_images:
                img_path = os.path.join(args.test_path, img_name)
                img = cv2.imread(img_path)
                
                if img is None:
                    continue

                # Ejecutamos el detector sobre la imagen
                detecciones = detector.detectar(img)
                
                # Para cada panel detectado, guardamos y dibujamos
                for det in detecciones:
                    x1, y1, x2, y2, score = det

                    
                    # Escribir en resultado.txt: nombre;x1;y1;x2;y2;1;score
                    linea = f"{img_name};{int(x1)};{int(y1)};{int(x2)};{int(y2)};1;{score:.2f}\n"
                    f_res.write(linea)
                    
                    # Dibujar en la imagen para "resultado_imgs" 
                    # Rectángulo rojo y score en amarillo 
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(img, f"{score:.2f}", (int(x1), int(y1)-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # Guardar la imagen procesada en el directorio correspondiente 
                cv2.imwrite(os.path.join("resultado_imgs", img_name), img)





    print("Proceso finalizado. Resultados guardados en resultado.txt y resultado_imgs/")

    # Evaluate detections
