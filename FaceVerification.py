import threading
import os
import logging

import cv2
from deepface import DeepFace

user_id = input(">>Enter your ID: ")
VERIFICATION_IMAGE_PATH = os.path.join(
    "database", "verification_images", user_id + ".jpg"
)

# frame counter
counter = 0

face_match = False
# recognition model -> SFace
model_name = "SFace"
# Detector model -> use Yunet
backend_detector = "yunet"
# use metric as Euclidean L2 form from author's clain
distance_metric = "cosine"
# prevent exception when the face is not detected
enforce_detection = True
align = True
# Normalization technique -> base = no normalization
normalization = "base"
# result = {'facial_areas': {'img1': {'x': 0, 'y': 0, 'w': 0, 'h': 0}, 'img2': {'x': 0, 'y': 0, 'w': 0, 'h': 0}}}

reference_img = cv2.imread(VERIFICATION_IMAGE_PATH)
cv2.imshow("reference_image", reference_img)

logging.basicConfig(level=logging.DEBUG)
logging.info(f"User ID: {user_id}")
logging.info(f"Verification image path: {VERIFICATION_IMAGE_PATH}")
logging.info(f"Model name: {model_name}")

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def verify_face(frame):
    global face_match
    global result
    try:
        result = DeepFace.verify(frame,
                                reference_img.copy(), 
                                model_name=model_name,
                                detector_backend=backend_detector,
                                distance_metric=distance_metric,
                                enforce_detection=enforce_detection,
                                align=align,
                                normalization=normalization,
                                )
        # print(result)
        if result["verified"]:
            face_match = True
        else:
            face_match = False
    except ValueError:
        face_match = False
    logging.info('Face match!' if face_match else 'No face match!')



while capture.isOpened():
    ret, frame = capture.read()

    if ret:
        frame = cv2.flip(frame, 1)
        if counter % 30 == 0:
            try:
                threading.Thread(target=verify_face, args=(frame.copy(),)).start()
            except ValueError:
                pass
        counter += 1

    if face_match:
        cv2.putText(
            frame, "FACE MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3
        )
    else:
        cv2.putText(
            frame,
            "NO FACE MATCH!",
            (20, 450),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            3,
        )
    try:
        user_bbox = result['facial_areas']['img1']
        x, y, w, h = user_bbox['x'], user_bbox['y'], user_bbox['w'], user_bbox['h']
        cv2.rectangle(frame, 
                        (x, y), 
                        (x + w, y + h), 
                        (255, 0, 0), 
                        2
                    )
    except NameError:
        pass
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()
