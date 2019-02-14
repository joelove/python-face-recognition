import cv2
import time
import utilities.face_utility

def capture(screen):
    identifying = False
    video_capture = cv2.VideoCapture(0)

    while True:
        video_capture.grab()
        ret, frame = video_capture.retrieve()

        if (ret == False):
            break;

        image = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        faces = utilities.face_utility.identify_faces(image)

        screen.clear()

        if (not faces):
            screen.print_at('None', 2, 1)

        for index, (identifier, distance) in enumerate(faces):
            name = identifier if (distance < 0.6) else 'Unknown'
            screen.print_at(name, 2, 1 + index)

        screen.refresh()
        time.sleep(0.1)

    video_capture.release()
