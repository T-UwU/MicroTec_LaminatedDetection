# INE Lamination Detector

Detecta laminación adicional en fotografías de credenciales INE mexicanas utilizando técnicas de visión por computadora **sin deep learning**. Identifica cuando una credencial ha sido plastificada con una funda o mica protectora adicional sobre la laminación original de fábrica.

## Qué hace

El flujo completo de detección:

1. **Carga la imagen** y detecta automáticamente la región de la credencial
2. **Aplica 9 métodos de detección** que analizan bordes, texturas, brillos y gradientes
3. **Combina puntuaciones** mediante un sistema de pesos calibrado
4. **Aplica ajustes por patrones** para reducir falsos positivos
5. **Clasifica como LAMINADA o NO LAMINADA** basado en umbral optimizado

### Indicadores de laminación

El detector busca características visuales típicas de credenciales plastificadas:
- **Doble contorno** — Borde visible de la mica protectora adicional
- **Gradientes en bordes** — Transiciones de luz en el perímetro exterior
- **Brillos especulares** — Reflejos característicos de superficies plásticas
- **Uniformidad de textura** — Superficies más lisas por el plástico adicional

## Métodos de detección

El detector combina 9 métodos independientes que analizan diferentes características:

| Método | Peso | Descripción |
|--------|------|-------------|
| `double_contour` | 20% | Detecta bordes dobles paralelos característicos de laminación mediante análisis multi-escala de gradientes perpendiculares al contorno de la tarjeta |
| `image_borders` | 20% | Analiza los bordes visibles de la imagen buscando patrones de doble borde en las zonas periféricas |
| `border_gradient` | 15% | Mide la magnitud del gradiente en bandas concéntricas alrededor del contorno de la tarjeta |
| `glare` | 8% | Detecta áreas brillantes con baja saturación (brillos por flash o reflejo) |
| `edge_sharpness` | 8% | Analiza la nitidez de bordes usando varianza del Laplaciano |
| `specular` | 8% | Identifica puntos muy brillantes y pequeños típicos de superficies plásticas |
| `saturation` | 8% | Evalúa la saturación del color (tarjetas laminadas preservan mejor los colores) |
| `color_uniformity` | 8% | Analiza la distribución de tonos en la imagen |
| `texture` | 5% | Mide la uniformidad de textura (superficies laminadas son más lisas) |

### Ajustes por patrones

El detector aplica correcciones basadas en patrones observados:
- **Alto brillo + baja saturación** → Reduce puntuación (probable flash, no laminación)
- **Alto border_gradient + alto double_contour** → Aumenta puntuación (fuerte indicador)
- **Valores moderados en múltiples métricas** → Penaliza (patrón de falso positivo)
- **Alta saturación + double_contour moderado** → Ajuste positivo leve

## Requisitos

Python 3.8+ con las siguientes dependencias:

```bash
pip install opencv-python numpy
```

## Uso

### Imagen individual (línea de comandos)

```bash
python lamination_detector.py "/ruta/a/credencial.jpg"
```

### Ejecutar pruebas del dataset

```bash
python lamination_detector.py
```

### Uso programático

```python
from lamination_detector import ImprovedLaminationDetector

detector = ImprovedLaminationDetector(debug=False)
result = detector.detect_lamination("/ruta/a/imagen.jpg")

print(f"Resultado: {result['result']}")
print(f"Confianza: {result['confidence']*100:.1f}%")
print(f"Puntuación: {result['total_score']:.3f}")
```

## Parámetros

| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `image_path` | Ruta a la imagen de la credencial INE | Requerido |
| `visualize` | Activa modo debug con imágenes de análisis | `False` |
| `debug` | Almacena imágenes intermedias para depuración | `False` |

## Salidas

El detector retorna un diccionario JSON con la siguiente estructura:

```json
{
  "result": "LAMINATED",
  "confidence": 0.772,
  "total_score": 0.754,
  "threshold": 0.46,
  "method_scores": {
    "double_contour": 0.808,
    "glare": 0.0,
    "edge_sharpness": 1.0,
    "texture": 0.0,
    "specular": 0.0,
    "saturation": 0.254,
    "color_uniformity": 0.93,
    "border_gradient": 1.0,
    "image_borders": 0.708
  }
}
```

### Campos de salida

| Campo | Descripción |
|-------|-------------|
| `result` | Clasificación: `"LAMINATED"` o `"NOT LAMINATED"` |
| `confidence` | Nivel de confianza de la clasificación (0.0 - 1.0) |
| `total_score` | Puntuación ponderada total (umbral: 0.46) |
| `threshold` | Umbral de clasificación utilizado |
| `method_scores` | Puntuaciones individuales de cada método (0.0 - 1.0) |

## Resultados de pruebas

### Dataset de evaluación

Resultados en 10 imágenes de credenciales INE:

| Archivo | Clasificación | Confianza | Ground Truth | Resultado |
|---------|---------------|-----------|--------------|-----------|
| `image_2026-02-24_125417453.png` | LAMINADA | 61.8% | LAMINADA | ✓ |
| `image_2026-02-24_125425170.png` | LAMINADA | 77.2% | LAMINADA | ✓ |
| `image_2026-02-24_125431619.png` | NO LAMINADA | 61.0% | NO LAMINADA | ✓ |
| `image_2026-02-24_125437842.png` | LAMINADA | 51.0% | LAMINADA | ✓ |
| `image_2026-02-24_125448786.png` | NO LAMINADA | 54.2% | NO LAMINADA | ✓ |
| `image_2026-02-24_125454451.png` | LAMINADA | 77.2% | LAMINADA | ✓ |
| `image_2026-02-24_125500668.png` | NO LAMINADA | 58.4% | NO LAMINADA | ✓ |
| `image_2026-02-24_125510143.png` | LAMINADA | 60.7% | LAMINADA | ✓ |
| `image_2026-02-24_125533311.png` | LAMINADA | 72.4% | LAMINADA | ✓ |
| `image_2026-02-24_125541423.png` | LAMINADA | 50.8% | LAMINADA | ✓ |

#### Resumen de clasificaciones

| Clasificación | Cantidad | Porcentaje |
|---------------|----------|------------|
| LAMINADA | 7 | 70% |
| NO LAMINADA | 3 | 30% |

#### Estadísticas por método

| Método | Mínimo | Máximo | Promedio |
|--------|--------|--------|----------|
| double_contour | 0.544 | 0.912 | 0.693 |
| border_gradient | 0.130 | 1.000 | 0.545 |
| image_borders | 0.447 | 0.792 | 0.642 |
| edge_sharpness | 0.628 | 1.000 | 0.881 |
| saturation | 0.233 | 0.798 | 0.413 |
| color_uniformity | 0.157 | 1.000 | 0.699 |
| glare | 0.000 | 0.161 | 0.025 |
| specular | 0.000 | 0.043 | 0.008 |
| texture | 0.000 | 0.640 | 0.114 |

## Niveles de confianza

| Confianza | Significado |
|-----------|-------------|
| > 75% | **Alta** — Detección muy confiable, indicadores claros |
| 60-75% | **Media** — Detección confiable, algunos indicadores presentes |
| 50-60% | **Baja** — Detección marginal, cerca del umbral de decisión |
| < 50% | **Muy baja** — Clasificación incierta, revisar manualmente |

## Ejemplo de integración

```python
from lamination_detector import ImprovedLaminationDetector
import json

def verificar_credencial(ruta_imagen):
    """
    Verifica si una credencial INE tiene laminación adicional.
    
    Args:
        ruta_imagen: Ruta al archivo de imagen
        
    Returns:
        dict con resultado de verificación
    """
    detector = ImprovedLaminationDetector(debug=False)
    resultado = detector.detect_lamination(ruta_imagen)
    
    if resultado.get('error'):
        return {
            'valida': False,
            'mensaje': f"Error: {resultado['error']}"
        }
    
    es_laminada = resultado['result'] == 'LAMINATED'
    confianza = resultado['confidence']
    
    return {
        'laminada': es_laminada,
        'confianza': f"{confianza*100:.1f}%",
        'puntuacion': resultado['total_score'],
        'umbral': resultado['threshold'],
        'indicadores_clave': {
            'doble_contorno': resultado['method_scores']['double_contour'],
            'gradiente_borde': resultado['method_scores']['border_gradient'],
            'bordes_imagen': resultado['method_scores']['image_borders']
        }
    }


# Ejemplo de uso
if __name__ == '__main__':
    ruta = "/home/ubuntu/Uploads/credencial.jpg"
    resultado = verificar_credencial(ruta)
    
    if resultado['laminada']:
        print(f"⚠️ Credencial LAMINADA detectada")
        print(f"   Confianza: {resultado['confianza']}")
        print(f"   Indicadores:")
        for k, v in resultado['indicadores_clave'].items():
            print(f"     - {k}: {v:.3f}")
    else:
        print(f"✓ Credencial sin laminación adicional")
        print(f"   Confianza: {resultado['confianza']}")
```

## Procesamiento en lote

```python
from lamination_detector import ImprovedLaminationDetector
from pathlib import Path
import json

def procesar_lote(directorio, salida_json):
    """Procesa múltiples imágenes y guarda resultados."""
    detector = ImprovedLaminationDetector()
    resultados = []
    
    for img_path in Path(directorio).glob('*.{jpg,jpeg,png}'):
        resultado = detector.detect_lamination(str(img_path))
        resultados.append({
            'archivo': img_path.name,
            'clasificacion': resultado['result'],
            'confianza': resultado['confidence'],
            'puntuacion': resultado['total_score']
        })
    
    with open(salida_json, 'w') as f:
        json.dump(resultados, f, indent=2)
    
    # Resumen
    laminadas = sum(1 for r in resultados if r['clasificacion'] == 'LAMINATED')
    print(f"Procesadas: {len(resultados)} imágenes")
    print(f"Laminadas: {laminadas}")
    print(f"No laminadas: {len(resultados) - laminadas}")
    
    return resultados
```
