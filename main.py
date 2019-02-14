from asciimatics.screen import Screen

import webcam
import utilities.face_utility

def main(screen):
    utilities.face_utility.create_faces_file()
    webcam.capture(screen)

if __name__ == '__main__':
    Screen.wrapper(main)
