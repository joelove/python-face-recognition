import cv2
import sys
import os

from time import time
from threading import Thread
from utilities.face_utility import identify_face


def begin_capture(screen):
    identifying = False
    video_capture = cv2.VideoCapture(0)

    while True:
        video_capture.grab()

        if (not identifying):
            ret, frame = video_capture.retrieve()

            if (ret == False):
                break;

            identifying = True

            try:
                frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
                start = time()
                identifier, distance = identify_face(frame)
                screen.clear()

                if (distance < 0.6):
                    screen.print_at(identifier, 2, 1)
                    # distance, time() - start
                else:
                    screen.print_at('Unknown', 2, 1)

            except Exception as e:
                screen.print_at('None', 2, 1)

        screen.refresh()
        identifying = False

    video_capture.release()
