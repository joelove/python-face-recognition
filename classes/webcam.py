import cv2

from time import time
from threading import Thread

from utilities.logging_utility import log
from utilities.face_utility import identify_face


class Webcam:
    identifying = False

    def begin_capture():
        video_capture = cv2.VideoCapture(0)

        while True:
            video_capture.grab()

            if (not self.identifying):
                ret, frame = video_capture.retrieve()

                if (ret == False):
                    print('No frame')
                    break

                self.identifying = True
                thread = Thread(target = process_frame, args = (frame, cb))
                thread.daemon = True
                thread.start()

        video_capture.release()

    def process_frame(frame, cb):
        try:
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
            start = time()
            identifier, distance = identify(frame)

            if (distance < 0.6):
                log(identifier, distance, time() - start)
            else:
                log('-', distance, time() - start)

            sys.stdout.flush()

        except Exception as e:
            exc = e
            log(None, 0, time() - start)

        self.identifying = False
