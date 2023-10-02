# Kivy dependencies
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.logger import Logger

import os
import cv2
from deepface import DeepFace


# build the app
class FaceIDApp(App):
    def build(self):
        # Main layout components
        self.web_cam = Image(size_hint=(1, 0.8))
        self.button = Button(text="Verify", on_press=self.verify, size_hint=(1, 0.1))
        self.verification_label = Label(
            text="Verification Uninitiated", size_hint=(1, 0.1)
        )

        # Add items to layout
        layout = BoxLayout(orientation="vertical")
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        # Setup video capture device
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        Clock.schedule_interval(self.update, 1.0 / 33.0)
        return layout

    # Run continuously to get webcam feed
    def update(self, *args):
        # Read frame from opencv
        ret, self.frame = self.capture.read()

        # Flip horizontall and convert image to texture
        buf = cv2.flip(self.frame, 0).tostring()
        img_texture = Texture.create(
            size=(self.frame.shape[1], self.frame.shape[0]), colorfmt="bgr"
        )
        img_texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
        self.web_cam.texture = img_texture

    # Verification function
    def verify(self, *args):
        self.face_match = False
        model_name = "FaceNet"
        reference_image_path = os.path.join(
            "./database", "verification_images", "input_image", "reference.jpg"
        )
        reference_image = cv2.imread(reference_image_path)
        try:
            result = DeepFace.verify(
                self.frame,
                reference_image.copy(),
                model_name=model_name,
            )
            if result["verified"]:
                self.face_match = True
            else:
                self.face_match = False

        except ValueError:
            self.verification_label.text = "No face detected"
            self.face_match = False

        self.verification_label.text = "Face Match: " + str(self.face_match)

        # Logs
        Logger.info("Face Match: " + str(self.face_match))
        Logger.info("Face Match: " + str(result["verified"]))


if __name__ == "__main__":
    FaceIDApp().run()
