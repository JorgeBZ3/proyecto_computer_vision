"""
Asignatura: Visión por Ordenador
Proyecto: Gimnasio virtual - Sistema de seguridad

Módulo de detección de objetos de gimnasio. Cada objeto se identifica mediante
segmentación por color en el espacio HSV y, a continuación, se verifica que la
forma de la región segmentada sea coherente con el objeto esperado.
"""

import os
from typing import Optional, Tuple

import cv2
import numpy as np


# File where the calibrated HSV ranges are stored
CALIBRATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "hsv_ranges.npz")

# Default HSV ranges (OpenCV: H in [0, 179], S and V in [0, 255])
DEFAULT_HSV_RANGES = {
    "botella": (np.array([172, 150, 70]), np.array([6, 255, 255])),    # red (H min > H max)
    "mancuerna": (np.array([100, 120, 60]), np.array([125, 255, 255])),  # blue
    "pelota": (np.array([43, 80, 50]), np.array([85, 255, 255])),        # green
    "banda": (np.array([22, 100, 70]), np.array([33, 255, 255])),        # yellow
}

# BGR colors used to draw each object on screen
DRAW_COLORS = {
    "botella": (0, 0, 255),
    "mancuerna": (255, 0, 0),
    "pelota": (0, 255, 0),
    "banda": (0, 255, 255),
}

# Minimum area of a single blob (fraction of the ROI area) to be considered
MIN_BLOB_RATIO = 0.003
# Minimum total area of an object (fraction of the ROI area)
MIN_OBJECT_RATIO = 0.02


def load_hsv_ranges(path: str = CALIBRATION_FILE) -> dict:
    """Carga los rangos HSV calibrados o, si no existen, los rangos por defecto.

    Args:
        path: Ruta del fichero .npz generado por la herramienta de calibración.

    Returns:
        Diccionario {nombre_objeto: (límite_inferior, límite_superior)}.
    """
    hsv_ranges = {}
    for name in DEFAULT_HSV_RANGES:
        hsv_ranges[name] = DEFAULT_HSV_RANGES[name]

    if os.path.exists(path):
        data = np.load(path)
        for name in hsv_ranges:
            if f"{name}_lower" in data.files and f"{name}_upper" in data.files:
                hsv_ranges[name] = (data[f"{name}_lower"], data[f"{name}_upper"])

    return hsv_ranges


def save_hsv_ranges(hsv_ranges: dict, path: str = CALIBRATION_FILE):
    """Guarda los rangos HSV en un fichero .npz.

    Args:
        hsv_ranges: Diccionario {nombre_objeto: (límite_inferior, límite_superior)}.
        path: Ruta del fichero de salida.
    """
    arrays = {}
    for name in hsv_ranges:
        lower, upper = hsv_ranges[name]
        arrays[f"{name}_lower"] = np.array(lower, dtype=np.uint8)
        arrays[f"{name}_upper"] = np.array(upper, dtype=np.uint8)
    np.savez(path, **arrays)


def get_color_mask(hsv_img: np.array, lower: np.array, upper: np.array) -> np.array:
    """Obtiene la máscara binaria de un color y la limpia con operadores morfológicos.

    Si el límite inferior de H es mayor que el superior (por ejemplo, el rojo,
    que está en los dos extremos del canal H), se unen dos rangos.

    Args:
        hsv_img: Imagen en espacio HSV.
        lower: Límite inferior (H, S, V).
        upper: Límite superior (H, S, V).

    Returns:
        Máscara binaria (0 o 255) del color.
    """
    # cv2.inRange needs both limits with the same type
    lower = np.array(lower, dtype=np.uint8)
    upper = np.array(upper, dtype=np.uint8)

    if lower[0] <= upper[0]:
        mask = cv2.inRange(hsv_img, lower, upper)
    else:
        # Hue range wraps around 0 (e.g. red)
        mask_low = cv2.inRange(hsv_img, np.array([0, lower[1], lower[2]], dtype=np.uint8), upper)
        mask_high = cv2.inRange(hsv_img, lower, np.array([179, upper[1], upper[2]], dtype=np.uint8))
        mask = cv2.bitwise_or(mask_low, mask_high)

    # Opening removes small noise, closing fills small holes (e.g. reflections)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def get_object_points(mask: np.array, min_blob_area: float) -> Tuple[Optional[np.array], float, float]:
    """Agrupa todas las regiones significativas de una máscara en un único objeto.

    Se agrupan las regiones porque un objeto puede aparecer dividido en varias
    manchas (por ejemplo, los dos discos de la mancuerna o una mano tapando
    parte de la botella).

    Args:
        mask: Máscara binaria de un color.
        min_blob_area: Área mínima (en píxeles) de una región para no ser ruido.

    Returns:
        Tupla (puntos, área_total, perímetro_total). Los puntos son None si no
        hay ninguna región significativa.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    total_area = 0.0
    total_perimeter = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_blob_area:
            valid_contours.append(contour)
            total_area += area
            total_perimeter += cv2.arcLength(contour, True)

    if len(valid_contours) == 0:
        return None, 0.0, 0.0

    points = np.vstack(valid_contours)
    return points, total_area, total_perimeter


def compute_shape_features(points: np.array, area: float, perimeter: float) -> dict:
    """Calcula los rasgos de forma usados para verificar cada objeto.

    Args:
        points: Puntos de los contornos del objeto.
        area: Área total del objeto.
        perimeter: Perímetro total del objeto.

    Returns:
        Diccionario con:
            - elongation: lado largo / lado corto del rectángulo mínimo
              (invariante a rotación).
            - solidity: área / área de la envolvente convexa.
            - circularity: 4·pi·A / P² (1 para un círculo perfecto).
    """
    # Minimum rotated rectangle -> rotation invariant elongation
    (_, _), (rect_w, rect_h), _ = cv2.minAreaRect(points)
    long_side = max(rect_w, rect_h)
    short_side = max(min(rect_w, rect_h), 1.0)
    elongation = long_side / short_side

    hull = cv2.convexHull(points)
    hull_area = max(cv2.contourArea(hull), 1.0)
    solidity = area / hull_area

    circularity = 4 * np.pi * area / max(perimeter ** 2, 1.0)

    features = {
        "elongation": elongation,
        "solidity": solidity,
        "circularity": circularity,
    }
    return features


def check_shape(object_name: str, features: dict) -> bool:
    """Comprueba si los rasgos de forma son coherentes con el objeto.

    Args:
        object_name: Nombre del objeto asociado al color detectado.
        features: Rasgos calculados con compute_shape_features().

    Returns:
        True si la forma encaja con el objeto, False en caso contrario.
    """
    elongation = features["elongation"]
    solidity = features["solidity"]
    circularity = features["circularity"]

    if object_name == "botella":
        # Elongated and compact
        return elongation >= 1.8 and solidity >= 0.6
    if object_name == "mancuerna":
        # Elongated with a narrow bar in the middle -> low solidity
        return 1.4 <= elongation <= 5.0 and solidity <= 0.85
    if object_name == "pelota":
        # Round
        return elongation <= 1.4 and circularity >= 0.6
    if object_name == "banda":
        # Shown stretched -> very elongated
        return elongation >= 4.0
    return False


def detect_object(roi_img: np.array, hsv_ranges: dict,
                  check_shapes: bool = True) -> Tuple[Optional[str], Optional[np.array]]:
    """Detecta qué objeto de gimnasio aparece en la zona de presentación.

    Args:
        roi_img: Imagen BGR de la zona de presentación (ROI).
        hsv_ranges: Rangos HSV de cada objeto.
        check_shapes: Si es False, solo se usa el color (útil para depurar).

    Returns:
        Tupla (nombre_objeto, envolvente_convexa). Devuelve (None, None) si no
        hay ningún objeto o si hay más de un candidato (caso ambiguo).
    """
    roi_area = roi_img.shape[0] * roi_img.shape[1]
    min_blob_area = MIN_BLOB_RATIO * roi_area
    min_object_area = MIN_OBJECT_RATIO * roi_area

    blurred = cv2.GaussianBlur(roi_img, (5, 5), 0)
    hsv_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    candidates = []
    for name in hsv_ranges:
        lower, upper = hsv_ranges[name]
        mask = get_color_mask(hsv_img, lower, upper)
        points, area, perimeter = get_object_points(mask, min_blob_area)

        if points is None or area < min_object_area:
            continue

        if check_shapes:
            features = compute_shape_features(points, area, perimeter)
            if not check_shape(name, features):
                continue

        candidates.append((name, cv2.convexHull(points)))

    # Only accept the detection if exactly one object is present
    if len(candidates) != 1:
        return None, None
    return candidates[0]
