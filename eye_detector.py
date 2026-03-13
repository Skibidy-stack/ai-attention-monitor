# import cv2
# import mediapipe as mp
# import time
# import math
# import numpy as np
# import simpleaudio as sa
#
# # -------- SOUND SETUP --------
# frequency = 1000
# fs = 44100
# seconds = 0.3
# t = np.linspace(0, seconds, int(seconds * fs), False)
# tone = np.sin(frequency * t * 2 * np.pi)
# audio = (tone * 32767).astype(np.int16)
#
# def beep():
#     sa.play_buffer(audio, 1, 2, fs)
#
# # -------- MEDIAPIPE SETUP --------
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh()
#
# cap = cv2.VideoCapture(0)
#
# # Eye landmark indexes
# LEFT_EYE = [33,160,158,133,153,144]
#
# eye_closed_start = None
# red_triggered = False
#
# def distance(p1,p2):
#     return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
#
# while True:
#
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(rgb)
#
#     status = "GREEN"
#     color = (0,255,0)
#
#     if results.multi_face_landmarks:
#
#         for face_landmarks in results.multi_face_landmarks:
#
#             h, w, _ = frame.shape
#
#             xs = []
#             ys = []
#
#             eye_points = []
#
#             for idx in LEFT_EYE:
#
#                 lm = face_landmarks.landmark[idx]
#
#                 x = int(lm.x*w)
#                 y = int(lm.y*h)
#
#                 xs.append(x)
#                 ys.append(y)
#
#                 eye_points.append((x,y))
#
#                 cv2.circle(frame,(x,y),3,(0,255,0),-1)
#
#             # Draw face bounding box
#             for landmark in face_landmarks.landmark:
#                 xs.append(int(landmark.x*w))
#                 ys.append(int(landmark.y*h))
#
#             x_min = min(xs)
#             x_max = max(xs)
#             y_min = min(ys)
#             y_max = max(ys)
#
#             # -------- EAR CALCULATION --------
#             v1 = distance(eye_points[1],eye_points[5])
#             v2 = distance(eye_points[2],eye_points[4])
#             h_dist = distance(eye_points[0],eye_points[3])
#
#             ear = (v1 + v2) / (2.0 * h_dist)
#
#             cv2.putText(frame,f"EAR: {ear:.2f}",(30,90),
#                         cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
#
#             # -------- STATE LOGIC --------
#             if ear < 0.25:
#
#                 if eye_closed_start is None:
#                     eye_closed_start = time.time()
#
#                 closed_time = time.time() - eye_closed_start
#
#                 if closed_time >= 5:
#
#                     status = "RED ALERT"
#                     color = (0,0,255)
#
#                     if not red_triggered:
#                         beep()
#                         red_triggered = True
#
#                 elif closed_time >= 3:
#
#                     status = "YELLOW WARNING"
#                     color = (0,255,255)
#
#             else:
#
#                 eye_closed_start = None
#                 red_triggered = False
#
#             # Draw colored bounding box
#             cv2.rectangle(frame,(x_min,y_min),(x_max,y_max),color,3)
#
#     # -------- STATUS TEXT --------
#     cv2.putText(frame,status,(30,50),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 1,color,3)
#
#     cv2.imshow("AI Eye Monitoring System",frame)
#
#     if cv2.waitKey(1) & 0xFF == 27:
#         break
#
# cap.release()
# cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import numpy as np
import simpleaudio as sa

# -------- Sound Alert --------
def beep():
    frequency = 800
    fs = 44100
    seconds = 0.2

    t = np.linspace(0, seconds, int(seconds * fs), False)
    tone = np.sin(frequency * t * 2 * np.pi)

    audio = tone * (2**15 - 1) / np.max(np.abs(tone))
    audio = audio.astype(np.int16)

    play_obj = sa.play_buffer(audio, 1, 2, fs)
    play_obj.wait_done()


# -------- Mediapipe Setup --------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# -------- Eye Aspect Ratio --------
def ear(eye, landmarks, frame_w, frame_h):
    points = []

    for i in eye:
        x = int(landmarks[i].x * frame_w)
        y = int(landmarks[i].y * frame_h)
        points.append((x, y))

    A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))

    ear = (A + B) / (2.0 * C)
    return ear


# -------- Camera --------
cap = cv2.VideoCapture(0)

closed_frames = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_h, frame_w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        landmarks = results.multi_face_landmarks[0].landmark

        left_ear = ear(LEFT_EYE, landmarks, frame_w, frame_h)
        right_ear = ear(RIGHT_EYE, landmarks, frame_w, frame_h)

        avg_ear = (left_ear + right_ear) / 2

        # Green focus dot
        cv2.circle(frame, (40, 40), 10, (0, 255, 0), -1)

        if avg_ear < 0.23:
            closed_frames += 1
        else:
            closed_frames = 0

        # If eyes closed too long
        if closed_frames > 25:
            cv2.putText(
                frame,
                "WAKE UP!",
                (200, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3,
            )

            cv2.circle(frame, (40, 40), 10, (0, 0, 255), -1)

            beep()

    cv2.imshow("AI Eye Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()