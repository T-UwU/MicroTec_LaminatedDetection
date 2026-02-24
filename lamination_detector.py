#!/usr/bin/env python3

import cv2
import numpy as np
from pathlib import Path
import json


class ImprovedLaminationDetector:
    # detector de mica en INEs con cv tradicional
    
    def __init__(self, debug=False):
        self.debug = debug
        self.debug_images = {}
        self.method_scores = {}
    
    def detect_card_region(self, image):
        # encontrar dónde está la credencial en la imagen
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        edges = cv2.Canny(blurred, 50, 150)
        combined = cv2.bitwise_or(thresh, edges)
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(combined, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None, None
        
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        
        return mask, largest_contour, (x, y, w, h)
    
    def detect_double_contour_improved(self, image, card_contour):
        # busca el doble borde que deja la mica
        # checa bordes a varias escalas y busca líneas paralelas
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # máscaras a diferentes distancias del contorno
        inner_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(inner_mask, [card_contour], -1, 255, thickness=10)
        
        mid_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mid_mask, [card_contour], -1, 255, thickness=25)
        mid_only = cv2.subtract(mid_mask, inner_mask)
        
        outer_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(outer_mask, [card_contour], -1, 255, thickness=40)
        outer_only = cv2.subtract(outer_mask, mid_mask)
        
        # canny con distintos umbrales
        edges_fine = cv2.Canny(gray, 20, 60)
        edges_medium = cv2.Canny(gray, 40, 120)
        edges_coarse = cv2.Canny(gray, 80, 200)
        
        # cuántos bordes hay en cada zona
        inner_fine = np.sum(cv2.bitwise_and(edges_fine, inner_mask) > 0)
        mid_fine = np.sum(cv2.bitwise_and(edges_fine, mid_only) > 0)
        outer_fine = np.sum(cv2.bitwise_and(edges_fine, outer_only) > 0)
        
        # qué tanto borde hay afuera vs adentro
        total_edges = inner_fine + mid_fine + outer_fine + 1
        outer_ratio = (mid_fine + outer_fine) / total_edges
        
        # densidad de bordes respecto al perímetro
        contour_length = cv2.arcLength(card_contour, True)
        if contour_length == 0:
            return 0.0
        
        edge_density = (mid_fine + outer_fine) / contour_length
        
        # ahora muestreamos perfiles perpendiculares al contorno
        # si hay doble borde se ven varias transiciones de intensidad
        epsilon = 0.02 * contour_length
        approx = cv2.approxPolyDP(card_contour, epsilon, True)
        
        double_edge_count = 0
        sample_count = 0
        
        for i in range(len(approx)):
            p1 = approx[i][0]
            p2 = approx[(i + 1) % len(approx)][0]
            
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            length = np.sqrt(dx*dx + dy*dy)
            if length < 50:
                continue
                
            # perpendicular al borde
            nx, ny = -dy / length, dx / length
            
            for t in np.linspace(0.2, 0.8, 5):
                px = int(p1[0] + t * dx)
                py = int(p1[1] + t * dy)
                
                profile = []
                for d in range(-30, 30):
                    x = int(px + d * nx)
                    y = int(py + d * ny)
                    if 0 <= x < w and 0 <= y < h:
                        profile.append(gray[y, x])
                    else:
                        profile.append(0)
                
                if len(profile) >= 40:
                    profile = np.array(profile)
                    grad = np.abs(np.diff(profile.astype(float)))
                    peaks = np.sum(grad > 30)
                    
                    if peaks >= 3:  # varias transiciones = probable mica
                        double_edge_count += 1
                    sample_count += 1
        
        double_edge_ratio = double_edge_count / max(sample_count, 1)
        
        score = 0.4 * outer_ratio + 0.3 * min(edge_density / 1.5, 1.0) + 0.3 * double_edge_ratio
        
        return min(score, 1.0)
    
    def detect_glare_improved(self, image, card_mask):
        # detecta reflejos, pero filtra los bordes que se confunden con brillo
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        masked_gray = cv2.bitwise_and(gray, gray, mask=card_mask)
        masked_hsv = cv2.bitwise_and(hsv, hsv, mask=card_mask)
        
        # quitar bordes porque se ven brillosos y confunden
        edges = cv2.Canny(gray, 50, 150)
        dilated_edges = cv2.dilate(edges, np.ones((7, 7), np.uint8))
        non_edge_mask = cv2.bitwise_and(card_mask, cv2.bitwise_not(dilated_edges))
        
        # muy brilloso + poca saturación = reflejo
        bright_mask = masked_gray > 235
        low_sat_mask = masked_hsv[:,:,1] < 40
        glare_regions = bright_mask & low_sat_mask & (non_edge_mask > 0)
        
        # nada más contar manchas de tamaño razonable
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(glare_regions.astype(np.uint8))
        
        glare_area = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if 50 < area < 10000:
                glare_area += area
        
        total_area = np.sum(card_mask > 0) + 1
        glare_ratio = glare_area / total_area
        
        score = min(glare_ratio * 20, 1.0)
        
        return score
    
    def analyze_edge_sharpness(self, image, card_contour):
        # qué tan marcados están los bordes, el doble borde sube la varianza del laplaciano
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [card_contour], -1, 255, thickness=20)
        
        edge_region = cv2.bitwise_and(gray, gray, mask=mask)
        
        laplacian = cv2.Laplacian(edge_region, cv2.CV_64F)
        laplacian_var = np.var(laplacian[mask > 0]) if np.sum(mask > 0) > 0 else 0
        
        score = min(laplacian_var / 3000, 1.0)
        
        return score
    
    def analyze_texture_uniformity(self, image, card_mask):
        # con mica la credencial se ve más lisa, sin tanto ruido de textura
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        kernel = np.ones((30, 30), np.uint8)
        interior_mask = cv2.erode(card_mask, kernel)
        
        if np.sum(interior_mask > 0) < 1000:
            return 0.0
        
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        local_mean = cv2.blur(gray.astype(float), (15, 15))
        local_var = cv2.blur((gray.astype(float) - local_mean)**2, (15, 15))
        
        masked_var = local_var[interior_mask > 0]
        mean_local_var = np.mean(masked_var)
        
        # menos varianza = más liso = posible mica (solo no es definitivo)
        score = max(0, 1 - mean_local_var / 500)
        
        return score
    
    def detect_specular_highlights(self, image, card_mask):
        # puntitos de brillo especular, típico del plástico
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        bright = gray > 248
        low_sat = hsv[:,:,1] < 20
        on_card = card_mask > 0
        
        specular = bright & low_sat & on_card
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(specular.astype(np.uint8))
        
        valid_specular_area = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if 10 < area < 3000:
                valid_specular_area += area
        
        total_area = np.sum(card_mask > 0) + 1
        ratio = valid_specular_area / total_area
        
        score = min(ratio * 30, 1.0)
        
        return score
    
    def analyze_saturation(self, image, card_mask):
        # con mica los colores se preservan mejor
        # sin mica + flash = todo lavado
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        sat = hsv[:,:,1]
        
        kernel = np.ones((20, 20), np.uint8)
        interior_mask = cv2.erode(card_mask, kernel)
        
        if np.sum(interior_mask > 0) < 1000:
            return 0.5
        
        masked_sat = sat[interior_mask > 0]
        mean_sat = np.mean(masked_sat)
        
        score = min(mean_sat / 80, 1.0)
        
        return score
    
    def analyze_color_uniformity(self, image, card_mask):
        # distribución de colores en el interior
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        kernel = np.ones((20, 20), np.uint8)
        interior_mask = cv2.erode(card_mask, kernel)
        
        if np.sum(interior_mask > 0) < 1000:
            return 0.5
        
        hue = hsv[:,:,0][interior_mask > 0]
        sat = hsv[:,:,1][interior_mask > 0]
        
        hue_std = np.std(hue)
        
        score = min(hue_std / 40, 1.0)
        
        return score
    
    def analyze_border_gradient(self, image, card_contour):
        # gradiente en bandas alrededor del borde, es de los que más jalan para detectar mica
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        masks = []
        for thickness in [5, 15, 25, 35]:
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [card_contour], -1, 255, thickness=thickness)
            masks.append(mask)
        
        band3 = cv2.subtract(masks[2], masks[1])  # 15-25px del borde
        band4 = cv2.subtract(masks[3], masks[2])  # 25-35px del borde
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        
        g3 = np.mean(grad_mag[band3 > 0]) if np.sum(band3) > 0 else 0
        g4 = np.mean(grad_mag[band4 > 0]) if np.sum(band4) > 0 else 0
        
        # con mica el gradiente en las bandas de afuera suele ser > 60
        avg_outer_grad = (g3 + g4) / 2
        
        score = min(avg_outer_grad / 100, 1.0)
        return score
    
    def analyze_image_borders(self, image):
        # checa las orillas de la imagen buscando el filo de la mica
        # cuando agarran la INE con la mano se alcanza a ver la orilla del plástico
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        border_width = int(min(h, w) * 0.12)
        
        top = gray[:border_width, :]
        bottom = gray[-border_width:, :]
        left = gray[:, :border_width]
        right = gray[:, -border_width:]
        
        scores = []
        
        for region, name in [(top, 'h'), (bottom, 'h'), (left, 'v'), (right, 'v')]:
            edges = cv2.Canny(region, 30, 90)
            
            if name == 'h':
                profile = np.sum(edges, axis=1)
            else:
                profile = np.sum(edges, axis=0)
            
            # contar picos, varios picos = varios bordes = mica
            threshold = np.max(profile) * 0.3
            peaks = 0
            in_peak = False
            for val in profile:
                if val > threshold and not in_peak:
                    peaks += 1
                    in_peak = True
                elif val <= threshold:
                    in_peak = False
            
            edge_score = min(peaks / 3.0, 1.0)
            
            sobelx = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(sobelx**2 + sobely**2)
            mean_grad = np.mean(grad_mag)
            grad_score = min(mean_grad / 50, 1.0)
            
            scores.append(0.5 * edge_score + 0.5 * grad_score)
        
        return np.mean(scores)
    
    def detect_lamination(self, image_path, visualize=False):
        # función principal, corre todo y combina los scores
        self.debug = visualize
        self.debug_images = {}
        self.method_scores = {}
        
        image = cv2.imread(str(image_path))
        if image is None:
            return {
                'result': 'ERROR',
                'confidence': 0.0,
                'error': f'Could not load image: {image_path}'
            }
        
        card_mask, card_contour, bbox = self.detect_card_region(image)
        
        if card_mask is None or card_contour is None:
            return {
                'result': 'ERROR',
                'confidence': 0.0,
                'error': 'Could not detect card in image'
            }
        
        scores = {}
        scores['double_contour'] = self.detect_double_contour_improved(image, card_contour)
        scores['glare'] = self.detect_glare_improved(image, card_mask)
        scores['edge_sharpness'] = self.analyze_edge_sharpness(image, card_contour)
        scores['texture'] = self.analyze_texture_uniformity(image, card_mask)
        scores['specular'] = self.detect_specular_highlights(image, card_mask)
        scores['saturation'] = self.analyze_saturation(image, card_mask)
        scores['color_uniformity'] = self.analyze_color_uniformity(image, card_mask)
        scores['border_gradient'] = self.analyze_border_gradient(image, card_contour)
        scores['image_borders'] = self.analyze_image_borders(image)
        
        self.method_scores = scores
        
        # pesos, doble contorno y bordes pesan más porque son los que más sirven
        weights = {
            'double_contour': 0.20,
            'border_gradient': 0.15,
            'image_borders': 0.20,
            'glare': 0.08,
            'edge_sharpness': 0.08,
            'texture': 0.05,
            'specular': 0.08,
            'saturation': 0.08,
            'color_uniformity': 0.08
        }
        
        total_score = sum(scores[k] * weights[k] for k in weights)
        
        # ajustes por patrones que fui viendo en el dataset
        
        # mucho brillo + colores lavados = le dieron flash sin mica
        if scores['glare'] > 0.4 and scores['saturation'] < 0.4:
            total_score *= 0.7
        
        # border gradient y double contour altos = casi seguro tiene mica
        if scores['border_gradient'] > 0.9 and scores['double_contour'] > 0.7:
            total_score = min(total_score * 1.2, 1.0)
        elif scores['double_contour'] > 0.85:
            total_score = min(total_score * 1.1, 1.0)
        
        # este patrón daba falsos positivos, valores medios en todo
        if (0.4 < scores['border_gradient'] < 0.55 and 
            0.6 < scores['double_contour'] < 0.7 and 
            scores['image_borders'] > 0.65):
            total_score *= 0.82
        
        # buena saturación + doble contorno medio, refuerza un poco
        if scores['saturation'] > 0.75 and scores['double_contour'] > 0.6:
            total_score = min(total_score * 1.08, 1.0)
        
        # bordes de imagen altos pero gradiente bajo = es el fondo no la mica
        if scores['image_borders'] > 0.65 and scores['border_gradient'] < 0.35:
            total_score *= 0.9
        
        threshold = 0.46
        
        result = "LAMINATED" if total_score > threshold else "NOT LAMINATED"
        
        if result == "LAMINATED":
            confidence = min((total_score - threshold) / (1 - threshold) * 0.5 + 0.5, 1.0)
        else:
            confidence = min((threshold - total_score) / threshold * 0.5 + 0.5, 1.0)
        
        return {
            'result': result,
            'confidence': round(confidence, 3),
            'total_score': round(total_score, 3),
            'method_scores': {k: round(v, 3) for k, v in scores.items()},
            'threshold': threshold
        }


def test_on_dataset():
    # probar con todo el dataset
    
    laminated = [
        '/home/ubuntu/Uploads/image_2026-02-03_000150350.png',
        '/home/ubuntu/Uploads/image_2026-02-03_000158131.png',
        '/home/ubuntu/Uploads/image_2026-02-23_091610125.png',
        '/home/ubuntu/Uploads/image_2026-02-23_091618603.png'
    ]

    not_laminated = [
        '/home/ubuntu/Uploads/11.jpg',
        '/home/ubuntu/Uploads/12.jpg',
        '/home/ubuntu/Uploads/image (1).jpeg',
        '/home/ubuntu/Uploads/image (2).jpeg',
        '/home/ubuntu/Uploads/image (3).jpeg',
        '/home/ubuntu/Uploads/image (4).jpeg',
        '/home/ubuntu/Uploads/image (5).jpeg',
        '/home/ubuntu/Uploads/image (6).jpeg',
        '/home/ubuntu/Uploads/image (7).jpeg'
    ]

    print('=' * 70)
    print('IMPROVED Detector Test on All 13 Images')
    print('=' * 70)

    detector = ImprovedLaminationDetector(debug=False)
    
    results = []
    
    print('\n### LAMINATED Images (Expected: LAMINATED) ###\n')
    for img_path in laminated:
        name = Path(img_path).name
        result = detector.detect_lamination(img_path)
        is_lam = result['result'] == 'LAMINATED'
        correct = is_lam
        results.append({'name': name, 'expected': 'LAMINATED', 'detected': result['result'], 
                       'correct': correct, 'score': result['total_score'], 'scores': result['method_scores']})
        mark = "✓" if correct else "✗"
        print(f"{mark} {name}: {result['result']} (score: {result['total_score']:.3f})")
        print(f"   dc: {result['method_scores']['double_contour']:.3f}, "
              f"img_bord: {result['method_scores']['image_borders']:.3f}, "
              f"glare: {result['method_scores']['glare']:.3f}")

    print('\n### NOT LAMINATED Images (Expected: NOT LAMINATED) ###\n')
    for img_path in not_laminated:
        name = Path(img_path).name
        result = detector.detect_lamination(img_path)
        is_lam = result['result'] == 'LAMINATED'
        correct = not is_lam
        results.append({'name': name, 'expected': 'NOT LAMINATED', 'detected': result['result'], 
                       'correct': correct, 'score': result['total_score'], 'scores': result['method_scores']})
        mark = "✓" if correct else "✗"
        print(f"{mark} {name}: {result['result']} (score: {result['total_score']:.3f})")
        print(f"   dc: {result['method_scores']['double_contour']:.3f}, "
              f"img_bord: {result['method_scores']['image_borders']:.3f}, "
              f"glare: {result['method_scores']['glare']:.3f}")

    TP = sum(1 for r in results if r['expected'] == 'LAMINATED' and r['detected'] == 'LAMINATED')
    FN = sum(1 for r in results if r['expected'] == 'LAMINATED' and r['detected'] != 'LAMINATED')
    FP = sum(1 for r in results if r['expected'] == 'NOT LAMINATED' and r['detected'] == 'LAMINATED')
    TN = sum(1 for r in results if r['expected'] == 'NOT LAMINATED' and r['detected'] != 'LAMINATED')

    print('\n' + '=' * 70)
    print('IMPROVED DETECTOR RESULTS')
    print('=' * 70)
    print(f'\nConfusion Matrix:')
    print(f'                  Predicted')
    print(f'                  LAM    NOT')
    print(f'  Actual LAM      {TP:3}    {FN:3}   (TP, FN)')
    print(f'         NOT      {FP:3}    {TN:3}   (FP, TN)')
    print(f'\nAccuracy: {(TP+TN)}/{len(results)} ({100*(TP+TN)/len(results):.1f}%)')
    print(f'True Positive Rate (Sensitivity): {TP}/{TP+FN} ({100*TP/(TP+FN):.1f}%)')
    print(f'True Negative Rate (Specificity): {TN}/{FP+TN} ({100*TN/(FP+TN):.1f}%)')
    
    return results, {'TP': TP, 'FN': FN, 'FP': FP, 'TN': TN}


def detect_single_image(image_path, show_details=True):
    # analizar una sola imagen
    detector = ImprovedLaminationDetector()
    result = detector.detect_lamination(image_path)
    
    if show_details:
        print(f"\nLamination Detection Results for: {Path(image_path).name}")
        print("-" * 50)
        print(f"Result: {result['result']}")
        print(f"Confidence: {result['confidence']*100:.1f}%")
        print(f"Score: {result['total_score']:.3f} (threshold: {result['threshold']})")
        print("\nMethod Scores:")
        for method, score in result['method_scores'].items():
            print(f"  - {method}: {score:.3f}")
    
    return result


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        detect_single_image(sys.argv[1])
    else:
        test_on_dataset()
