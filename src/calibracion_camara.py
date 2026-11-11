"""
Calibración de la cámara del ordenador con un damero (chessboard).

Flujo:
  1. Se abre la cámara en directo (basado en el test.py del profesor).
  2. Con ESPACIO (o 'c') se guarda la imagen actual en la carpeta 'calibration/'
     (solo si se detecta el damero completo).
  3. Con 'q' o ESC se termina la captura y se calibra con todas las imágenes
     de 'calibration/', siguiendo los pasos de la práctica (lab_2, Apartado A):
       - load_images()
       - cv2.findChessboardCorners() + cv2.cornerSubPix()
       - cv2.drawChessboardCorners() (se guardan en calibration/corners_refined)
       - get_chessboard_points()
       - cv2.calibrateCamera()
  4. Se guardan los parámetros en 'calibration/calib_camera.npz'.

Uso:
  python calibracion_camara.py              # captura + calibración
  python calibracion_camara.py --solo-calibrar   # calibra con las imágenes ya guardadas
"""

import argparse
import copy
import glob
import os
from typing import List

import cv2
import numpy as np

# ------------------------------------------------------------------
# CONFIGURACIÓN (ajústala a tu damero)
# ------------------------------------------------------------------
# Número de esquinas INTERIORES del damero (columnas, filas).
# Un damero de 9x7 cuadrados tiene 8x6 esquinas interiores.
CHESSBOARD_SHAPE = (9, 6)
# Lado de cada cuadrado en mm (mídelo en tu damero impreso).
SQUARE_SIZE = 25

CALIB_FOLDER = "calibration"
CORNERS_FOLDER = os.path.join(CALIB_FOLDER, "corners_refined")
OUTPUT_FILE = os.path.join(CALIB_FOLDER, "calib_camera.npz")

CAMERA_INDEX = 0
WIDTH, HEIGHT = 1280, 720

# Criterio de refinamiento subpíxel (igual que en la práctica)
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)



def load_images(filenames: List) -> List:
    return [cv2.imread(img) for img in filenames]


def write_image(output_folder: str, img_name: str, img: np.array):
    os.makedirs(output_folder, exist_ok=True)
    img_path = os.path.join(output_folder, img_name)
    cv2.imwrite(img_path, img)


def get_chessboard_points(chessboard_shape, dx, dy):
    n_long, n_short = chessboard_shape

    puntos = []
    for i in range(n_short):
        for j in range(n_long):
            x = i * dx
            y = j * dy
            z = 0
            puntos.append([x, y, z])

    puntos = np.array(puntos, dtype=np.float32)
    return puntos


# ------------------------------------------------------------------
# 1) Captura de imágenes del damero
# ------------------------------------------------------------------
def next_image_index(folder: str) -> int:
    """Devuelve el siguiente número libre para no sobrescribir capturas previas."""
    existing = glob.glob(os.path.join(folder, "img_*.png"))
    if not existing:
        return 0
    nums = [int(os.path.basename(p)[4:7]) for p in existing]
    return max(nums) + 1


def detect_preview(frame, scale=0.5):
    """Detección rápida (sobre imagen reducida) solo para la vista previa."""
    small = cv2.resize(frame, None, fx=scale, fy=scale)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray, CHESSBOARD_SHAPE, flags=cv2.CALIB_CB_FAST_CHECK
    )
    if found:
        corners = corners / scale
    return found, corners


def capture_images(camera_index=CAMERA_INDEX, width=WIDTH, height=HEIGHT):
    os.makedirs(CALIB_FOLDER, exist_ok=True)

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # CAP_DSHOW para Windows
    if not cap.isOpened():
        print(f"No se pudo abrir la cámara (índice {camera_index}).")
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    window_name = "Calibracion - ESPACIO: capturar | q/ESC: calibrar"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    idx = next_image_index(CALIB_FOLDER)
    n_captured = 0

    print("ESPACIO o 'c' -> capturar imagen | 'q' o ESC -> terminar y calibrar")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No se recibe imagen (¿cámara desconectada?).")
                break

            # Detección para la vista previa (sobre la imagen ORIGINAL)
            found, corners = detect_preview(frame)
            display = frame.copy()
            if found:
                cv2.drawChessboardCorners(display, CHESSBOARD_SHAPE,
                                          corners.astype(np.float32), True)

            # Espejo SOLO para mostrar; las imágenes guardadas NO se voltean,
            # porque una imagen espejada no corresponde a una cámara real.
            display = cv2.flip(display, 1)

            color = (0, 200, 0) if found else (0, 0, 255)
            estado = "Damero detectado" if found else "Damero NO detectado"
            cv2.putText(display, estado, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(display, f"Capturas: {n_captured}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)


            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord(' '), ord('c')):
                if found:
                    name = f"img_{idx:03d}.png"
                    write_image(CALIB_FOLDER, name, frame)
                    print(f"Guardada {os.path.join(CALIB_FOLDER, name)}")
                    idx += 1
                    n_captured += 1

                else:
                    print("No se detecta el damero completo, imagen no guardada.")
            elif key in (ord('q'), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Capturadas {n_captured} imágenes nuevas.")
    return True



def calibrate():
    # A.1 Cargar imágenes
    imgs_path = sorted(glob.glob(os.path.join(CALIB_FOLDER, "img_*.png")))
    if len(imgs_path) < 3:
        print(f"Hay {len(imgs_path)} imágenes en '{CALIB_FOLDER}'. "
              "Se necesitan al menos 3 (recomendado 15-20).")
        return None
    imgs = load_images(imgs_path)

    # A.2 Detectar esquinas y refinar con cornerSubPix
    corners = [cv2.findChessboardCorners(img, CHESSBOARD_SHAPE) for img in imgs]
    corners_copy = copy.deepcopy(corners)

    imgs_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]
    corners_refined = [
        cv2.cornerSubPix(img_gray, corners=corner[1], winSize=(5, 5),
                         zeroZone=(-1, -1), criteria=CRITERIA)
        if corner[0] else None
        for img_gray, corner in zip(imgs_gray, corners_copy)
    ]

    # A.3 Dibujar y guardar detecciones para comprobarlas
    imgs_copy = copy.deepcopy(imgs)
    for path, img, corner in zip(imgs_path, imgs_copy, corners_refined):
        if corner is not None:
            cv2.drawChessboardCorners(img, CHESSBOARD_SHAPE, corner, True)
            name = os.path.basename(path).replace("img_", "refined_")
            write_image(CORNERS_FOLDER, name, img)

    # Filtrar detecciones válidas
    valid_corners = [corner for corner in corners_refined if corner is not None]
    print(f"Imágenes con detección: {len(valid_corners)} de {len(imgs)}")
    if len(valid_corners) < 3:
        print("No hay suficientes detecciones válidas para calibrar.")
        return None
    valid_corners = np.array(valid_corners, dtype=np.float32)

    # A.4 Puntos 3D del damero (uno por imagen válida)
    chessboard_points = [get_chessboard_points(CHESSBOARD_SHAPE, SQUARE_SIZE, SQUARE_SIZE)
                         for _ in valid_corners]

    # A.5 Calibración
    image_size = imgs_gray[0].shape[::-1]  # (ancho, alto)
    rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        chessboard_points, valid_corners, imageSize=image_size,
        cameraMatrix=None, distCoeffs=None)

    # Extrínsecos
    extrinsics = list(map(lambda rvec, tvec: np.hstack((cv2.Rodrigues(rvec)[0], tvec)),
                          rvecs, tvecs))

    print("Intrinsics:\n", intrinsics)
    print("Distortion coefficients:\n", dist_coeffs)
    print("Root mean squared reprojection error:\n", rms)

    np.savez(OUTPUT_FILE, intrinsics=intrinsics, dist_coeffs=dist_coeffs, rms=rms,
             rvecs=rvecs, tvecs=tvecs, extrinsics=extrinsics,
             image_size=np.array(image_size))
    print(f"Parámetros guardados en {OUTPUT_FILE}")
    return intrinsics, dist_coeffs



def show_undistorted(intrinsics, dist_coeffs, camera_index=CAMERA_INDEX,
                     width=WIDTH, height=HEIGHT):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    window_name = "Original | Sin distorsion - q/ESC para salir"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            undistorted = cv2.undistort(frame, intrinsics, dist_coeffs)
            both = np.hstack((cv2.flip(frame, 1), cv2.flip(undistorted, 1)))
            cv2.imshow(window_name, both)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Calibración de cámara con damero")
    parser.add_argument("--solo-calibrar", action="store_true",
                        help="No abre la cámara; calibra con las imágenes ya guardadas")
    parser.add_argument("--camara", type=int, default=CAMERA_INDEX,
                        help="Índice de la cámara (por defecto 0)")
    args = parser.parse_args()

    if not args.solo_calibrar:
        if not capture_images(camera_index=args.camara):
            return

    result = calibrate()

    if result is not None and not args.solo_calibrar:
        resp = input("¿Ver la imagen corregida en directo? (s/n): ").strip().lower()
        if resp == "s":
            show_undistorted(*result, camera_index=args.camara)


if __name__ == "__main__":
    main()
