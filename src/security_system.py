"""
Bucle principal del sistema de seguridad. Lee la cámara, detecta el objeto que
se muestra en el recuadro, actualiza el decodificador y dibuja la interfaz.
Para juntarlo con el sistema principal basta con llamar a run_security_system()
pasándole la misma cámara.
"""

from typing import List, Optional

import cv2
import numpy as np

from color_detector import DRAW_COLORS, detect_object, load_hsv_ranges
from sequence_decoder import create_decoder, reset_decoder, update_decoder

import time 
from DrawModule import draw_fps


# Password: change this list to modify it.
# The objects are: botella, mancuerna and banda
DEFAULT_PASSWORD = ["banda", "banda", "banda", "banda"]


def get_time() -> float:
    """Devuelve el instante actual en segundos con el reloj de OpenCV.

    Returns:
        Tiempo en segundos.
    """
    return cv2.getTickCount() / cv2.getTickFrequency()


def get_roi_coords(frame: np.array, roi_ratio: float = 0.5) -> List[int]:
    """Calcula el recuadro de presentación centrado en la imagen.

    Args:
        frame: Imagen de la cámara.
        roi_ratio: Fracción del alto y del ancho que ocupa el recuadro.

    Returns:
        Coordenadas [x1, y1, x2, y2] del recuadro.
    """
    h, w = frame.shape[0:2]
    roi_w, roi_h = int(w * roi_ratio), int(h * roi_ratio)
    x1, y1 = (w - roi_w) // 2, (h - roi_h) // 2
    return [x1, y1, x1 + roi_w, y1 + roi_h]


def draw_progress(frame: np.array, decoder: dict):
    """Dibuja una casilla por cada objeto de la secuencia en la parte inferior.

    Args:
        frame: Imagen sobre la que se dibuja.
        decoder: Diccionario del decodificador.
    """
    h, w = frame.shape[0:2]
    box_size, gap = 50, 15
    n = decoder["sequence_length"]
    x0 = (w - (n * box_size + (n - 1) * gap)) // 2
    y0 = h - box_size - 20

    for i in range(n):
        x = x0 + i * (box_size + gap)
        if i < len(decoder["sequence"]):
            # Box filled with the color of the registered object
            color = DRAW_COLORS[decoder["sequence"][i]]
            cv2.rectangle(frame, (x, y0), (x + box_size, y0 + box_size), color, -1)
        cv2.rectangle(frame, (x, y0), (x + box_size, y0 + box_size), (255, 255, 255), 2)


def draw_interface(frame: np.array, roi: List[int], decoder: dict, detected: Optional[str],
                   hull: Optional[np.array], now: float):
    """Dibuja el recuadro, el objeto detectado, el mensaje de estado y el progreso.

    Args:
        frame: Imagen sobre la que se dibuja.
        roi: Coordenadas [x1, y1, x2, y2] del recuadro.
        decoder: Diccionario del decodificador.
        detected: Objeto detectado en el frame, o None.
        hull: Envolvente convexa del objeto (coordenadas del recuadro), o None.
        now: Instante actual en segundos.
    """
    x1, y1, x2, y2 = roi
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    if detected is not None:
        # Move the hull from ROI coordinates to frame coordinates
        cv2.drawContours(frame, [hull + np.array([x1, y1])], -1, DRAW_COLORS[detected], 3)
        cv2.putText(frame, detected, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    state = decoder["state"]
    if state == "GRANTED":
        message = "ACCESO CONCEDIDO"
    elif state == "DENIED":
        message = "SECUENCIA INCORRECTA (pulsa r)"
    elif state == "LOCKED":
        remaining = int(decoder["lockout_s"] - (now - decoder["lock_start"])) + 1
        message = f"BLOQUEADO ({remaining} s)"
    elif state == "RELEASE":
        message = "Retira el objeto"
    else:
        message = "Muestra un objeto en el recuadro"

    cv2.putText(frame, message, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(frame, "r: reset   q: salir", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    draw_progress(frame, decoder)


def run_security_system(cap: cv2.VideoCapture, window_name: str = "Gimnasio virtual") -> bool:
    """Ejecuta el sistema de seguridad hasta que se concede el acceso o se sale.

    No libera la cámara al terminar para que el sistema principal pueda seguir usándola.

    Args:
        cap: Cámara ya abierta con cv2.VideoCapture.
        window_name: Nombre de la ventana.

    Returns:
        True si se ha introducido la secuencia correcta, False si se sale con 'q'.

    Raises:
        RuntimeError: Si la cámara no está abierta.
    """
    if not cap.isOpened():
        raise RuntimeError("La cámara no está abierta")

    hsv_ranges = load_hsv_ranges()
    decoder = create_decoder(DEFAULT_PASSWORD)
    granted_time = None

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)  # mirror effect
        now = get_time()

        x1, y1, x2, y2 = get_roi_coords(frame)
        detected, hull = detect_object(frame[y1:y2, x1:x2], hsv_ranges)
        state = update_decoder(decoder, detected, now)

        draw_interface(frame, [x1, y1, x2, y2], decoder, detected, hull, now)

        prev_time = draw_fps(frame, prev_time)
        
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        # Keep the "access granted" message for a moment before leaving
        if state == "GRANTED":
            if granted_time is None:
                granted_time = now
            elif now - granted_time >= 1.5:
                cv2.destroyWindow(window_name)
                cv2.waitKey(1)  # let OpenCV process the close event
                return True

        if key == ord("q"):
            return False
        # The reset key does not skip the lockout
        if key == ord("r") and state != "LOCKED":
            reset_decoder(decoder)


def main():
    """Ejecuta el sistema de seguridad por separado."""
    cap = cv2.VideoCapture(0)
    granted = run_security_system(cap)
    print("Acceso concedido" if granted else "Acceso no concedido")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
