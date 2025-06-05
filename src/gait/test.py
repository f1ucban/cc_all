import cv2 as cv
import mediapipe as mp
from pathlib import Path

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
drawing_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)

video = Path("../cc_all/dataset/raw/s002/fv/civ/s002_civ1_fv_cp_nb_v1.mp4")
cap = cv.VideoCapture(str(video))

if not cap.isOpened():
    print("Error opening video file")
    exit()

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    smooth_landmarks=True,
) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(80, 110, 10)),
            mp_drawing.DrawingSpec(color=(80, 256, 121)),
        )

        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10)),
            mp_drawing.DrawingSpec(color=(80, 44, 121)),
        )

        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76)),
            mp_drawing.DrawingSpec(color=(121, 44, 250)),
        )

        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66)),
            mp_drawing.DrawingSpec(color=(245, 66, 230)),
        )

        cv.imshow("MediaPipe Detection", image)

        if cv.waitKey(25) & 0xFF == ord("q"):
            break

cap.release()
cv.destroyAllWindows()
