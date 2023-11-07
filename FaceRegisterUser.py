import os
import cv2
import logging


user_id = input(">>Enter your ID: ")
VERIFICATION_IMAGE_PATH = os.path.join(
    "database", "verification_images", user_id + ".jpg"
)

logging.basicConfig(level=logging.DEBUG)
logging.info(f"User ID: {user_id}")
logging.info(f"Verification image path: {VERIFICATION_IMAGE_PATH}")

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = capture.read()
    if ret:
        frame = cv2.flip(frame, 1)
        cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.imwrite(VERIFICATION_IMAGE_PATH, frame)
        logging.info("Verification image saved")
        break

capture.release()
cv2.destroyAllWindows()
