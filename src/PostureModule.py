"""
Clase para detectar la pose de un cuerpo de una camara en directo, y extraer la posicion de partes del cuerpo.
Se puede extraer las coordenadas de los puntos corporales en la lista de landmarks (lm_list),
con 33 puntos cada uno perteneciendo a una parte del cuerpo.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import math

class poseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True,
                  detectionCon=0.5, trackCon=0.5):
        self.mode = mode    # Falso para que primero detecte postura, y luego solo la siga.
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.lm_list = []       # Coordenadas en pixeles (para dibujar)
        self.world_list = []    # Coordenadas 3D en metros (para calcular angulos)


        self.mpDraw = mp.solutions.drawing_utils # Para dibujar puntos
        self.mpPose = mp.solutions.pose  # Acceder al modulo de deteccion de postura
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=1,     # 0: mas rapido, 1: mas preciso (confunde menos brazos y piernas)
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )    # Objeto de deteccion de postura
        self.results = None  # Donde se guarda la postura que ha detectado el modelo.

    def find_pose(self, frame, draw=True):
        # Metodo que detecta la postura de la persona, y lo guarda en results.
        # Convertir a RGB para que sea compatible con MediaPipe
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Pasar la imagen al modelo, guardar el resultado, y luego pintarla.
        self.results = self.pose.process(frameRGB)

        if self.results.pose_landmarks: # Si se ha detectado un cuerpo
            if draw:
                self.mpDraw.draw_landmarks(frame, self.results.pose_landmarks,
                                        self.mpPose.POSE_CONNECTIONS)
        return frame


    def find_position(self, frame, draw=True):
        # Metodo que extrae las coordenadas de los puntos corporales que el modelo
        # ya ha detectado accediendo a results.
        # Si quieres dibujar todos los puntos, draw=True.
        # Actualiza self.lm_list, pero tambien se puede obtener a traves del return.
        self.lm_list = []   # Se vacia en cada frame: si no hay cuerpo, la lista queda vacia
        self.world_list = []
        if self.results and self.results.pose_landmarks: # Si se ha detectado un cuerpo
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x*w), int(lm.y*h)  # Convertir x,y de ratio de imagen a pixel
                self.lm_list.append([id,cx,cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 5, (255,0,0), cv2.FILLED)  # Ponerle un circulo para asegurar que son las coordenadas correctas
            # Mismos puntos en 3D (metros, con el origen entre las caderas)
            for lm in self.results.pose_world_landmarks.landmark:
                self.world_list.append([lm.x, lm.y, lm.z])
        return self.lm_list

    def find_angle(self, frame, p1, p2, p3, draw=True, use_3d=True):
        # Devuelve el angulo (en grados, 0-180) que forman los puntos p1-p2-p3, con el vertice en p2.
        # use_3d=True: con world_list (3D), sale igual de frente o de perfil, pero depende
        #              de la z que estima el modelo.
        # use_3d=False: con lm_list (2D), el angulo tal como se ve en la imagen.
        if use_3d:
            P1, P2, P3 = self.world_list[p1], self.world_list[p2], self.world_list[p3]
        else:
            P1, P2, P3 = self.lm_list[p1][1:], self.lm_list[p2][1:], self.lm_list[p3][1:]
        a = np.array(P1) - np.array(P2)
        b = np.array(P3) - np.array(P2)
        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
        angle = math.degrees(math.acos(np.clip(cos, -1, 1)))

        # Puntos en pixeles para dibujar
        x1, y1 = self.lm_list[p1][1:]
        x2, y2 = self.lm_list[p2][1:]
        x3, y3 = self.lm_list[p3][1:]


        # Dibujar
        if draw:
            cv2.line(frame, (x1,y1), (x2,y2), (255,255,255), 3)
            cv2.line(frame, (x2,y2), (x3,y3), (255,255,255), 3)

            cv2.circle(frame, (x1, y1), 5, (255,0,0), cv2.FILLED)
            cv2.circle(frame, (x1, y1), 15, (0,0,255), 2)
            cv2.circle(frame, (x2, y2), 5, (255,0,0), cv2.FILLED)
            cv2.circle(frame, (x2, y2), 15, (0,0,255), 2)
            cv2.circle(frame, (x3, y3), 5, (255,0,0), cv2.FILLED)
            cv2.circle(frame, (x3, y3), 15, (0,0,255), 2)
            cv2.putText(frame, str(int(angle)), (x2-50, y2+50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

        return angle





def main():
    # Read video
    capture = cv2.VideoCapture(0)    # 0 para capturar la camara

    t_prev = 0   # Tiempo inicial para calcular el frame rate
    detector = poseDetector()
    draw_body = False   # Para quitar o poner diagrama de puntos sobre el cuerpo.
    while True:
        ret, frame = capture.read()
        if not ret:
            break

        frame = detector.find_pose(frame, draw=draw_body)   # draw=False para que solo se dibuje la parte del cuerpo que yo quiera∫
        lm_list = detector.find_position(frame, draw=False)
        if len(lm_list) != 0:   # Por si no detecta un cuerpo es ese frame
            cv2.circle(frame, (lm_list[13][1], lm_list[13][2]), 15, (255,0,0), cv2.FILLED)  # 13 = Left elbow
            detector.find_angle(frame, 11, 13, 15)  # Puntos del brazo izquierdo
            detector.find_angle(frame, 12, 14, 16)

        # FPS
        t_new = time.time()
        fps = 1/(t_new-t_prev)
        t_prev = t_new
        cv2.putText(frame, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255,0,0), 3)

        cv2.imshow("Video", frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            cv2.destroyAllWindows()
            break
        elif key == ord("d"):
            draw_body = not draw_body


if __name__ == "__main__":
    main()