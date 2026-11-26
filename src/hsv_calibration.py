"""
Asignatura: Visión por Ordenador
Proyecto: Gimnasio virtual - Sistema de seguridad

Herramienta de calibración de los rangos HSV. Para cada objeto se ajustan los
límites con barras deslizantes viendo la máscara en tiempo real, y al final
se guardan en un fichero .npz que carga el sistema de seguridad.

Controles: 's' guarda el objeto actual y pasa al siguiente, 'n' lo salta
sin cambios y 'q' sale sin guardar.
"""

import cv2
import numpy as np

from color_detector import get_color_mask, load_hsv_ranges, save_hsv_ranges


WINDOW_NAME = "Calibracion HSV"
TRACKBARS = ["H min", "H max", "S min", "S max", "V min", "V max"]
TRACKBAR_MAX = [179, 179, 255, 255, 255, 255]


def nothing(value: int):
    """Callback vacío requerido por cv2.createTrackbar().

    Args:
        value: Valor de la barra (no se usa).
    """
    pass


def create_trackbars(lower: np.array, upper: np.array):
    """Crea las barras deslizantes con los valores iniciales de un objeto.

    Args:
        lower: Límite inferior inicial (H, S, V).
        upper: Límite superior inicial (H, S, V).
    """
    initial_values = [lower[0], upper[0], lower[1], upper[1], lower[2], upper[2]]
    for i in range(len(TRACKBARS)):
        cv2.createTrackbar(TRACKBARS[i], WINDOW_NAME, int(initial_values[i]), TRACKBAR_MAX[i], nothing)


def read_trackbars() -> tuple:
    """Lee los valores actuales de las barras deslizantes.

    Returns:
        Tupla (límite_inferior, límite_superior) como arrays de numpy.
    """
    values = []
    for name in TRACKBARS:
        values.append(cv2.getTrackbarPos(name, WINDOW_NAME))
    lower = np.array([values[0], values[2], values[4]], dtype=np.uint8)
    upper = np.array([values[1], values[3], values[5]], dtype=np.uint8)
    return lower, upper


def calibrate_object(cap: cv2.VideoCapture, name: str, lower: np.array, upper: np.array) -> tuple:
    """Calibra el rango HSV de un objeto de forma interactiva.

    Args:
        cap: Cámara abierta.
        name: Nombre del objeto que se calibra.
        lower: Límite inferior inicial.
        upper: Límite superior inicial.

    Returns:
        Tupla (límite_inferior, límite_superior, tecla_pulsada).
    """
    cv2.destroyAllWindows()
    cv2.namedWindow(WINDOW_NAME)
    create_trackbars(lower, upper)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)

        lower, upper = read_trackbars()
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = get_color_mask(hsv_img, lower, upper)
        segmented = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.putText(frame, f"Objeto: {name}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, "s: guardar  n: saltar  q: salir", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Show original and segmented images side by side
        combined = np.hstack((frame, segmented))
        cv2.imshow(WINDOW_NAME, combined)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("s"), ord("n"), ord("q")):
            return lower, upper, key


def main():
    """Recorre todos los objetos, calibra sus rangos y los guarda."""
    cap = cv2.VideoCapture(0)
    hsv_ranges = load_hsv_ranges()

    for name in hsv_ranges:
        lower, upper = hsv_ranges[name]
        new_lower, new_upper, key = calibrate_object(cap, name, lower, upper)

        if key == ord("q"):
            print("Calibración cancelada, no se guarda nada")
            cap.release()
            cv2.destroyAllWindows()
            return
        if key == ord("s"):
            hsv_ranges[name] = (new_lower, new_upper)
            print(f"{name}: lower = {new_lower}, upper = {new_upper}")

    save_hsv_ranges(hsv_ranges)
    print("Rangos guardados")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
