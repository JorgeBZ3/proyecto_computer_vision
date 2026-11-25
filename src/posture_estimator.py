"""
Fichero para detectar la pose de un cuerpo de una camara en directo.
"""

import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils # Para dibujar puntos
mpPose = mp.solutions.pose  # Acceder al modulo de deteccion de postura
pose = mpPose.Pose()    # Objeto de deteccion de postura


# Read video
capture = cv2.VideoCapture(0)    # 0 para capturar la camara 

t_prev = 0   # Tiempo inicial para calcular el frame rate
while True:
    ret, frame = capture.read()
    if not ret:
        break

    # Convertir a RGB para que sea compatible con MediaPipe
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Pasar la imagen al modelo, guardar el resultado, y luego pintarla.
    results = pose.process(frameRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = frame.shape
            cx, cy = int(lm.x*w), int(lm.y*h)  # Convertir x,y de ratio de imagen a pixel
            #cv2.circle(frame, (cx, cy), 5, (255,0,0), cv2.FILLED)  # Ponerle un circulo para asegurar que son las coordenadas correctas

    # FPS
    t_new = time.time()
    fps = 1/(t_new-t_prev)
    t_prev = t_new
    cv2.putText(frame, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255,0,0), 3)
    
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) == "q":
        break
