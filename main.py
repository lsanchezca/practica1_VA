import cv2
import numpy as np
import os
import argparse

class PanelDetector:
    def __init__(self):
        # Ajustar parámetros de MSER para evitar ruido
        self.mser = cv2.MSER_create(min_area=2000, max_area=100000)

    def calcular_score(self, roi):
        # 1. Pasar el recorte a HSV 
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 2. Definir el rango del azul en HSV (ajustar tras ver imágenes de train)
        # El azul suele estar entre 100-130 en OpenCV Hue
        lower_blue = np.array([100, 100, 50])
        upper_blue = np.array([130, 255, 255])
        
        # 3. Crear máscara: píxeles azules = 1, resto = 0
        mask = cv2.inRange(hsv_roi, lower_blue, upper_blue)
        mask = (mask > 0).astype(np.float32)
        
        # 4. Redimensionar a tamaño fijo (p.ej. 40x80) para correlar
        mask_resized = cv2.resize(mask, (80, 40))
        
        # 5. Máscara ideal: un panel de carretera es casi todo azul
        ideal_mask = np.ones((40, 80), dtype=np.float32)
        
        # 6. Correlación (multiplicar elemento a elemento y sumar)
        # Esto da la proporción de píxeles azules en el área
        score = np.sum(mask_resized * ideal_mask) / (40 * 80)
        
        return float(score)

    def non_maximum_suppression(self, boxes):
        # Sección 3: Algoritmo para eliminar ventanas repetidas
        # Por ahora devolvemos las mismas cajas hasta que lo implementemos
        return boxes

    def detectar(self, image):
        # Convertir a gris para MSER
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        regions, _ = self.mser.detectRegions(gray)
        
        detecciones = []
        for p in regions:
            x, y, w, h = cv2.boundingRect(p)
            
            # Filtrado por relación de aspecto
            aspect_ratio = w / float(h)
            if 0.5 < aspect_ratio < 4.0:
                # Agrandar rectángulo para incluir borde blanco
                # x1, y1, x2, y2, score
                score = self.calcular_score(image[y:y+h, x:x+w])
                if score > 0.4:
                    detecciones.append([x, y, x+w, y+h, score])
        
        return self.non_maximum_suppression(detecciones)


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
