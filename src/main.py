"""
AI Trainer: menu principal.
  1 -> Modo basico (curl de biceps, flexion, sentadilla)
  2 -> Modo combinado (sentadilla + flexion)
  3 -> Modo competicion (2 jugadores, 1 minuto cada uno)
  q / ESC -> Salir

Uso:
  python main.py
"""

import cv2

import PostureModule as pm
import modos
from DrawModule import WINDOW_NAME, draw_screen

# ------------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------------
SISTEMA_OPERATIVO = "windows"   # "mac", "windows" u otro (linux)
CAMERA_INDEX = 0
WIDTH, HEIGHT = 1280, 720

MENU = ["AI TRAINER",
        "",
        "1 - Modo basico",
        "2 - Modo combinado (sentadilla + flexion)",
        "3 - Modo competicion (2 jugadores)",
        "q - Salir"]


def open_camera(camera_index=CAMERA_INDEX, width=WIDTH, height=HEIGHT):
    if SISTEMA_OPERATIVO == "windows":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # CAP_DSHOW para Windows
    elif SISTEMA_OPERATIVO == "mac":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(camera_index)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def main():
    capture = open_camera()
    if not capture.isOpened():
        print(f"No se pudo abrir la cámara (índice {CAMERA_INDEX}).")
        return

    detector = pm.poseDetector()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = modos.read_frame(capture)
            if frame is None:
                print("No se recibe imagen (¿cámara desconectada?).")
                break

            frame = draw_screen(frame, MENU)
            cv2.imshow(WINDOW_NAME, frame)

            # Leer la opcion del menu
            key = cv2.waitKey(1) & 0xFF
            if key == ord("1"):
                modos.modo_basico(capture, detector)
            elif key == ord("2"):
                modos.modo_combinado(capture, detector)
            elif key == ord("3"):
                modos.modo_competicion(capture, detector)
            elif key in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()