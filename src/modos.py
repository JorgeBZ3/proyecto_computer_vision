"""
Los tres modos del programa. Cada modo tiene su propio bucle de video y
vuelve al menu al pulsar 'q' o ESC.
  1. modo_basico:      un ejercicio a elegir (curl, flexion, sentadilla).
  2. modo_combinado:   sentadilla + flexion = 1 repeticion combinada.
  3. modo_competicion: 2 jugadores a la vez (cada uno en una mitad de la imagen), 1 minuto,
                       gana quien haga mas repeticiones.
"""

import cv2
import time
import numpy as np

import PostureModule as pm
from ExerciseModule import EJERCICIOS, repCounter, exercise_from_key
from DrawModule import WINDOW_NAME, draw_text, draw_bar, draw_count, draw_screen, draw_fps

SECUENCIA_COMBINADA = ["sentadilla", "flexion"]   # Orden del ejercicio combinado
TIEMPO_COMPETICION = 60     # Segundos que dura la competicion
CUENTA_ATRAS = 3            # Segundos antes de empezar cada turno


def read_frame(capture):
    # Lee un frame de la camara en modo espejo. Devuelve None si no hay imagen.
    ret, frame = capture.read()
    if not ret:
        return None
    return cv2.flip(frame, 1)


def track(detector, frame, counter, draw_body=False):
    # Detecta la postura y actualiza el contador si hay un cuerpo en la imagen.
    # Devuelve el frame y True si se acaba de completar una repeticion.
    frame = detector.find_pose(frame, draw=draw_body)
    lm_list = detector.find_position(frame, draw=False)
    if len(lm_list) != 0:
        return frame, counter.update(detector, frame)
    draw_text(frame, "No se detecta ningun cuerpo", (20, 100))
    return frame, False


# 1) Modo basico
def modo_basico(capture, detector):
    exercise = "curl"
    counter = repCounter(exercise)
    draw_body = False   # Para quitar o poner diagrama de puntos sobre el cuerpo.

    prev_time = time.time()

    while True:
        frame = read_frame(capture)
        if frame is None:
            break

        frame, rep = track(detector, frame, counter, draw_body)

        draw_text(frame, EJERCICIOS[exercise]["nombre"], (20, 50), 1.5, thickness=3)
        draw_text(frame, "c/f/s: ejercicio | r: reiniciar | d: esqueleto | q: menu",
                  (280, frame.shape[0] - 20), 0.7)
        draw_bar(frame, counter.per, counter.dir)
        draw_count(frame, counter.count)

        prev_time = draw_fps(frame, prev_time)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        new_exercise = exercise_from_key(key)
        if new_exercise is not None:     # Cambiar de ejercicio
            exercise = new_exercise
            counter = repCounter(exercise)
        elif key == ord("r"):
            counter = repCounter(exercise)
        elif key == ord("d"):
            draw_body = not draw_body
        elif key in (ord("q"), 27):
            break


# 2) Modo combinado: sentadilla + flexion
def modo_combinado(capture, detector):
    fase = 0            # Ejercicio de la secuencia que toca hacer
    counter = repCounter(SECUENCIA_COMBINADA[fase])
    total = 0           # Repeticiones combinadas completas
    draw_body = False

    prev_time = time.time()

    while True:
        frame = read_frame(capture)
        if frame is None:
            break

        frame, rep = track(detector, frame, counter, draw_body)
        if rep:
            # Ejercicio terminado: pasar al siguiente de la secuencia
            fase = (fase + 1) % len(SECUENCIA_COMBINADA)
            if fase == 0:
                total += 1
            counter = repCounter(SECUENCIA_COMBINADA[fase])

        draw_text(frame, EJERCICIOS[counter.exercise]["nombre"], (20, 50), 1.5, thickness=3)
        draw_text(frame, "r: reiniciar | d: esqueleto | q: menu", (280, frame.shape[0] - 20), 0.7)
        draw_bar(frame, counter.per, counter.dir)
        draw_count(frame, total)

        prev_time = draw_fps(frame, prev_time)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            fase = 0
            counter = repCounter(SECUENCIA_COMBINADA[fase])
            total = 0
        elif key == ord("d"):
            draw_body = not draw_body
        elif key in (ord("q"), 27):
            break


# 3) Modo competicion: 2 jugadores a la vez
def modo_competicion(capture, detector):
    # MediaPipe Pose solo detecta una persona por imagen, asi que la imagen se parte en dos
    # mitades: el jugador 1 a la izquierda y el jugador 2 a la derecha, cada uno con su detector.
    detectores = [detector, pm.poseDetector()]
    exercise = "sentadilla"
    estado = "inicio"   # inicio -> cuenta_atras -> jugando -> resultado
    counters = [repCounter(exercise), repCounter(exercise)]
    t_inicio = 0

    prev_time = time.time()
    
    while True:
        frame = read_frame(capture)
        if frame is None:
            break

        h, w, c = frame.shape
        mitad = w // 2
        nombre = EJERCICIOS[exercise]["nombre"]
        t = time.time() - t_inicio

        if estado == "inicio":
            frame = draw_screen(frame, ["COMPETICION: " + nombre,
                                        "Jugador 1 a la izquierda, jugador 2 a la derecha",
                                        "c / f / s: cambiar ejercicio",
                                        f"Teneis {TIEMPO_COMPETICION} segundos",
                                        "",
                                        "ESPACIO: empezar | q: menu"])

        elif estado == "cuenta_atras":
            if t >= CUENTA_ATRAS:
                estado = "jugando"
                counters = [repCounter(exercise), repCounter(exercise)]
                t_inicio = time.time()
            else:
                draw_text(frame, str(CUENTA_ATRAS - int(t)), (w // 2 - 70, h // 2 + 80), 8, (0, 255, 255), 15)

        elif estado == "jugando":
            restante = TIEMPO_COMPETICION - t
            for i in range(2):
                # Cada jugador en su mitad (al dibujar en la mitad se dibuja en el frame)
                zona = frame[:, i * mitad:(i + 1) * mitad]
                track(detectores[i], zona, counters[i])
                draw_text(zona, f"Jugador {i + 1}   {int(np.ceil(restante))} s", (20, 50), 1.5, thickness=3)
                draw_bar(zona, counters[i].per, counters[i].dir)
                draw_count(zona, counters[i].count)

            if restante <= 0:   # Se acaba el tiempo
                estado = "resultado"

        elif estado == "resultado":
            puntos = [counters[0].count, counters[1].count]
            if puntos[0] > puntos[1]:
                ganador = "GANA EL JUGADOR 1"
            elif puntos[1] > puntos[0]:
                ganador = "GANA EL JUGADOR 2"
            else:
                ganador = "EMPATE"
            frame = draw_screen(frame, ["RESULTADO",
                                        f"Jugador 1: {puntos[0]} repeticiones",
                                        f"Jugador 2: {puntos[1]} repeticiones",
                                        ganador, "",
                                        "r: revancha | q: menu"])

        # Linea que separa a los dos jugadores
        if estado == "jugando":
            cv2.line(frame, (mitad, 0), (mitad, h), (255, 255, 255), 2)

        prev_time = draw_fps(frame, prev_time)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif estado == "inicio" and key == ord(" "):
            estado = "cuenta_atras"
            t_inicio = time.time()
        elif estado == "inicio" and exercise_from_key(key) is not None:
            exercise = exercise_from_key(key)
        elif estado == "resultado" and key == ord("r"):
            estado = "inicio"