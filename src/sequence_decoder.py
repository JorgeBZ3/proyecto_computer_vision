"""
Asignatura: Visión por Ordenador
Proyecto: Gimnasio virtual - Sistema de seguridad

Decodificador de secuencias. Memoriza hasta 4 objetos consecutivos y concede
o bloquea el acceso según si coinciden con la contraseña. No depende de la
cámara: en cada frame recibe el objeto detectado (o None) y el instante
actual, así que se puede probar con secuencias simuladas.

Estados posibles:
    - "WAITING": esperando un objeto nuevo.
    - "RELEASE": objeto registrado, esperando a que se retire.
    - "GRANTED": secuencia correcta.
    - "DENIED": secuencia incorrecta (espera a que se pulse la tecla de reset).
    - "LOCKED": bloqueado tras demasiados intentos fallidos.
"""

from typing import List, Optional


def create_decoder(password: List[str], sequence_length: int = 4, frames_to_confirm: int = 15,
                   frames_to_release: int = 10, max_attempts: int = 3, lockout_s: float = 15.0) -> dict:
    """Crea el diccionario con los parámetros y el estado del decodificador.

    Args:
        password: Secuencia correcta de objetos.
        sequence_length: Número de objetos de la secuencia.
        frames_to_confirm: Frames seguidos necesarios para registrar un objeto.
        frames_to_release: Frames seguidos sin objeto para aceptar el siguiente.
        max_attempts: Intentos fallidos permitidos antes del bloqueo.
        lockout_s: Duración del bloqueo en segundos.

    Returns:
        Diccionario con el estado del decodificador.
    """
    decoder = {
        "password": password,
        "sequence_length": sequence_length,
        "frames_to_confirm": frames_to_confirm,
        "frames_to_release": frames_to_release,
        "max_attempts": max_attempts,
        "lockout_s": lockout_s,
        "failed_attempts": 0,
        "lock_start": 0.0,
    }
    reset_decoder(decoder)
    return decoder


def reset_decoder(decoder: dict):
    """Olvida la secuencia actual y vuelve al estado de espera.

    Args:
        decoder: Diccionario del decodificador.
    """
    decoder["sequence"] = []
    decoder["state"] = "WAITING"
    decoder["candidate"] = None
    decoder["candidate_frames"] = 0
    decoder["empty_frames"] = 0


def check_sequence(decoder: dict, now: float):
    """Compara la secuencia completa con la contraseña.

    Solo se comprueba al final para no dar pistas de en qué posición se ha fallado.

    Args:
        decoder: Diccionario del decodificador.
        now: Instante actual en segundos.
    """
    if decoder["sequence"] == decoder["password"]:
        decoder["state"] = "GRANTED"
        decoder["failed_attempts"] = 0
        return

    decoder["failed_attempts"] += 1
    if decoder["failed_attempts"] >= decoder["max_attempts"]:
        decoder["state"] = "LOCKED"
        decoder["lock_start"] = now
    else:
        decoder["state"] = "DENIED"


def update_decoder(decoder: dict, detected: Optional[str], now: float) -> str:
    """Actualiza el decodificador con la detección de un frame.

    Args:
        decoder: Diccionario del decodificador.
        detected: Objeto detectado en el frame, o None si no hay ninguno.
        now: Instante actual en segundos.

    Returns:
        Estado del decodificador tras procesar el frame.
    """
    state = decoder["state"]

    if state == "GRANTED":
        return state

    if state == "LOCKED":
        if now - decoder["lock_start"] >= decoder["lockout_s"]:
            decoder["failed_attempts"] = 0
            reset_decoder(decoder)
        return decoder["state"]

    # A wrong sequence is only forgotten with the reset key
    if state == "DENIED":
        return state

    if decoder["state"] == "WAITING":
        if detected is None:
            decoder["candidate"] = None
            decoder["candidate_frames"] = 0
        elif detected == decoder["candidate"]:
            decoder["candidate_frames"] += 1
        else:
            decoder["candidate"] = detected
            decoder["candidate_frames"] = 1

        # Object seen for enough consecutive frames -> register it
        if decoder["candidate_frames"] >= decoder["frames_to_confirm"]:
            decoder["sequence"].append(detected)
            decoder["state"] = "RELEASE"
            decoder["empty_frames"] = 0
            if len(decoder["sequence"]) == decoder["sequence_length"]:
                check_sequence(decoder, now)

    elif decoder["state"] == "RELEASE":
        # Wait until the ROI is empty before accepting the next object
        if detected is None:
            decoder["empty_frames"] += 1
        else:
            decoder["empty_frames"] = 0

        if decoder["empty_frames"] >= decoder["frames_to_release"]:
            decoder["state"] = "WAITING"
            decoder["candidate"] = None
            decoder["candidate_frames"] = 0

    return decoder["state"]
