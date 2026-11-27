"""
Definicion de los ejercicios y del contador de repeticiones.
Una repeticion solo cuenta si se recorre el rango de movimiento:
posicion inicial (0%) -> rango completo (100%) -> vuelta a la posicion inicial (0%).
"""

import numpy as np

# ------------------------------------------------------------------
# CONFIGURACIÓN (ajústala a tu gusto)
# ------------------------------------------------------------------
# puntos:   (p1, p2, p3) del lado izquierdo y del derecho. El angulo se mide en p2.
# rango:    (angulo con el rango completo = 100%, angulo en la posicion inicial = 0%)
#           No hace falta estirar del todo: basta con pasar del angulo del 0%.
# posicion: postura del tronco necesaria para empezar/terminar la repeticion
#           ("vertical", "horizontal" o None si da igual)
# tecla:    tecla para elegir el ejercicio
EJERCICIOS = {
    "curl": {
        "nombre": "Curl de biceps",
        "puntos": [(11, 13, 15), (12, 14, 16)],   # hombro - codo - muñeca
        "rango": (50, 130),
        "posicion": None,
        "tecla": "c",
    },
    "flexion": {
        "nombre": "Flexion",
        "puntos": [(11, 13, 15), (12, 14, 16)],   # hombro - codo - muñeca
        "rango": (90, 140),
        "posicion": "horizontal",
        "tecla": "f",
    },
    "sentadilla": {
        "nombre": "Sentadilla",
        "puntos": [(23, 25, 27), (24, 26, 28)],   # cadera - rodilla - tobillo
        "rango": (100, 150),
        "posicion": "vertical",
        "tecla": "s",
    },
}


def exercise_from_key(key):
    # Devuelve el ejercicio asociado a la tecla pulsada (o None)
    for exercise, ej in EJERCICIOS.items():
        if key == ord(ej["tecla"]):
            return exercise
    return None


def get_percentage(detector, frame, exercise, draw=True):
    # Porcentaje del rango de movimiento (0-100%). Se calcula en los dos lados del
    # cuerpo y se usa el mas flexionado, asi vale cualquier brazo/pierna.
    ej = EJERCICIOS[exercise]
    per = 0
    for p1, p2, p3 in ej["puntos"]:
        angle = detector.find_angle(frame, p1, p2, p3, draw)
        per = max(per, np.interp(angle, ej["rango"], (100, 0)))
    return per


def check_position(world_list, posicion):
    # Comprueba en 3D si el tronco (caderas -> hombros) esta vertical u horizontal
    if posicion is None:
        return True
    hombros = (np.array(world_list[11]) + np.array(world_list[12])) / 2
    caderas = (np.array(world_list[23]) + np.array(world_list[24])) / 2
    dx, dy, dz = np.abs(hombros - caderas)
    vertical = dy > np.hypot(dx, dz)     # Mas altura que distancia en el suelo
    if posicion == "horizontal":
        return not vertical
    return vertical


class repCounter():
    def __init__(self, exercise):
        self.exercise = exercise
        self.count = 0
        self.per = 0
        # dir = -1: esperando la posicion inicial
        # dir =  0: en la posicion inicial, hay que llegar al 100%
        # dir =  1: rango completado, hay que volver a la posicion inicial
        self.dir = -1

    def update(self, detector, frame, draw=True):
        # Actualiza el contador con el frame actual. Devuelve True si se acaba de
        # completar una repeticion. Hay que llamar antes a detector.find_position().
        ej = EJERCICIOS[self.exercise]
        self.per = get_percentage(detector, frame, self.exercise, draw)

        if self.per >= 100 and self.dir == 0:
            self.dir = 1
        elif self.per <= 0 and check_position(detector.world_list, ej["posicion"]):
            if self.dir == 1:
                self.count += 1
                self.dir = 0
                return True
            self.dir = 0
        return False
