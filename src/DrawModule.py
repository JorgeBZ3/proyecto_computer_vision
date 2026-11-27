"""
Funciones para dibujar la interfaz sobre el frame.
(cv2.putText no pinta tildes ni la ñ, por eso los textos van sin ellas)
"""

import cv2
import numpy as np

WINDOW_NAME = "AI Trainer"
# Fuente de los textos. Otras opciones: cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_TRIPLEX,
# cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_PLAIN (la de antes, mas fina)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_text(frame, text, pos, scale=1, color=(255, 255, 255), thickness=2):
    cv2.putText(frame, text, pos, FONT, scale, color, thickness)


def draw_bar(frame, per, dir):
    # Barra del rango de movimiento. Se pone verde al llegar al 100% (toca volver).
    h, w, c = frame.shape
    color = (0, 255, 0) if dir == 1 else (255, 0, 255)
    bar = int(np.interp(per, (0, 100), (h - 100, 100)))
    cv2.rectangle(frame, (w - 100, 100), (w - 50, h - 100), color, 3)
    cv2.rectangle(frame, (w - 100, bar), (w - 50, h - 100), color, cv2.FILLED)
    draw_text(frame, f"{int(per)}%", (w - 115, 80), 1, color)


def draw_count(frame, count):
    # Numero de repeticiones abajo a la izquierda
    h, w, c = frame.shape
    cv2.rectangle(frame, (0, h - 150), (250, h), (0, 255, 0), cv2.FILLED)
    draw_text(frame, str(count), (30, h - 35), 3.5, (255, 0, 0), 8)


def draw_screen(frame, lines):
    # Oscurece el frame y escribe varias lineas de texto (menu, turnos, resultado)
    frame = frame // 3
    for i, line in enumerate(lines):
        draw_text(frame, line, (100, 150 + 70 * i), 1.3)
    return frame