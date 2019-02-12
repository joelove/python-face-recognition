import cv2

from time import time
from threading import Thread

from utilities.logging_utility import log
from utilities.face_utility import identify_face


identifying = False

def begin_capture():
    video_capture = cv2.VideoCapture(0)

    global identifying

    while True:
        video_capture.grab()

        if (not identifying):
            ret, frame = video_capture.retrieve()

            if (ret == False):
                break

            identifying = True
            thread = Thread(target = process_frame, args = (frame, log))
            thread.daemon = True
            thread.start()

    video_capture.release()

def process_frame(frame, log):
    try:
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        start = time()
        identifier, distance = identify_face(frame)

        if (distance < 0.6):
            log(identifier, distance, time() - start)
        else:
            log('-', distance, time() - start)

        sys.stdout.flush()

    except Exception as e:
        exc = e
        print(e)
        log(None, 0, time() - start)

    identifying = False
