import cv2
import logging
from gaze_tracking import GazeTracking
from tkinter import *

gaze = GazeTracking()
face_cascade = cv2.CascadeClassifier(r'.\haarcascades\haarcascade_frontalface_default.xml')
right_eye_cascade = cv2.CascadeClassifier(r'.\haarcascades\haarcascade_right_eye.xml')

pupil_detector = cv2.SimpleBlobDetector_create()
eyes = []
faces = []




def detectPupils(color_frame_full, preprocessed_frame_full, face, eye):
    (x, y, width, height) = face
    (ex, ey, ew, eh) = eye

    face_frame = preprocessed_frame_full[y: y + height, x: x + width]
    eye_frame = face_frame[ey: ey + eh, ex: ex + ew]

    preprocesses_eye_frame = preprocess_eye_frame(eye_frame)

    # Detect blobs
    keypoints = pupil_detector.detect(preprocesses_eye_frame)


    for keypoint in keypoints:
        pass

    return preprocessed_frame_full


def preprocess_eye_frame(eye_frame):
    # Apply adaptive thresholding
    max_output_value = 100
    neighorhood_size = 99
    subtract_from_mean = 8

    eye_frame = cv2.adaptiveThreshold(
        eye_frame,
        max_output_value,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        neighorhood_size,
        subtract_from_mean
        )

    return eye_frame




def detectEyes(color_frame_full, preprocessed_frame_full, face):
    (x, y, width, height) = face
    global eyes
    global faces
    faces = face
    face_frame = preprocessed_frame_full[y: y + height, x: x + width]

    right_eyes = right_eye_cascade.detectMultiScale(face_frame, 1.3, 12)
    eyes = right_eyes[:]
    for i,eye in enumerate(right_eyes):
        (ex, ey, ew, eh) = eye

        cv2.rectangle(
            face_frame,
            (ex, ey),
            (ex + ew, ey + eh),
            (0, 255, 0),
            2
            )

        cv2.putText(
            face_frame,
            f"Eye {i}",
            (ex, ey - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
            )

        #preprocessed_frame_full = detectPupils(color_frame_full, preprocessed_frame_full, face, eye)

    return preprocessed_frame_full


def detectFaces(color_frame_full, preprocessed_frame_full):
    faces = face_cascade.detectMultiScale(
        preprocessed_frame_full,
        1.3, # scale factor
        5 # minimum neighbors each candidate rectangle should have to retain it.
        )

    for face in faces:
        (x, y, width, height) = face
        # draw a rectangle in the color frame
        cv2.rectangle(
            preprocessed_frame_full,
            (x, y),
            (x + width, y + height),
            (255, 0, 0), # color of rectangle
            2 # thickness of rectangle
            )

        cv2.putText(
            preprocessed_frame_full,
            "Face",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
            )

        preprocessed_frame_full = detectEyes(color_frame_full, preprocessed_frame_full, face)

    return preprocessed_frame_full


def preprocess_color_frame_full(color_frame_full):
    preprocessed_frame_full = cv2.cvtColor(color_frame_full, cv2.COLOR_BGR2GRAY)

    # perform lighting processing here

    return preprocessed_frame_full


def start(color_frame_full):
    preprocessed_frame_full = preprocess_color_frame_full(color_frame_full)

    preprocessed_frame_full = detectFaces(color_frame_full, preprocessed_frame_full)

    return preprocessed_frame_full


def setup():
    # -- Setup logger --
    logging.basicConfig(
        format = '%(asctime)s %(levelname)-8s %(message)s',
    	level = logging.INFO,
		datefmt = '%H:%M:%S',
        filename = 'tracker.log',
        filemode = 'w')


    # -- Setup pupil detector --
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 30

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 35
    params.minArea = 100

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.4

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.5

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.3

    # Create a detector with the parameters
    global pupil_detector
    pupil_detector = cv2.SimpleBlobDetector_create(params)


def main():
    global eyes
    global faces
    # perform setup
    setup()

    # We turn the webcam on
    video_capture = cv2.VideoCapture(1)


    # get a frame
    while True:
        _, color_frame_full = video_capture.read()
        gaze.refresh(color_frame_full)

        # perform detection
        canvas = start(color_frame_full)
        # display frame with the rectangles
        color_frame_full = cv2.drawMarker(
            color_frame_full,
            gaze.pupil_left_coords(),
            (255, 255, 0),
            markerSize=10
        )
        color_frame_full = cv2.drawMarker(
            color_frame_full,
            gaze.pupil_right_coords(),
            (255, 255, 0),
            markerSize=10
        )
        pupils = [gaze.pupil_left_coords(), gaze.pupil_right_coords()]
        px_ratio = [0.5, 0.5]
        py_ratio = [0.5, 0.5]
        if len(faces):
            fx, fy, _, __ = faces
            if len(eyes) == 2:
                for i, eye in enumerate(eyes):
                     if pupils[i]:
                        (ex, ey, ew, eh) = eye
                        #print((ex, ey))
                        (px, py) = pupils[i]
                        color_frame_full = cv2.drawMarker(
                            color_frame_full,
                            (fx + ex, fy + ey),
                            (255, 255, 0),
                            markerSize=10
                        )
                        color_frame_full = cv2.drawMarker(
                            color_frame_full,
                            (fx + ex + ew, fy + ey+ eh),
                            (255, 255, 0),
                            markerSize=10
                        )
                        #print((ex, ey, px, py, ex+ew, ey+eh))
                        px_ratio[i] = (px - (fx+ex)) / ew
                        py_ratio[i] = (py - (fy+ey)) / eh

        cv2.imshow('Main Frame', canvas)
        cv2.imshow("Demo", color_frame_full)
        if cv2.waitKey(1) == 'q':
            break
    video_capture.release()

        #print(gaze.pupil_left_coords())
        #print(gaze.pupil_right_coords())

        # check for quit


    # We turn the webcam off.

    # close all windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()