import os
import numpy

from glob import glob
from skimage import io
from asciimatics.screen import Screen

import utilities.webcam_utility

from utilities.face_utility import face_to_vector, faces_from_image
from utilities.path_utility import build_path, name_from_path

def main(screen):
    if (not os.path.isfile('faces/faces')):
        analyzed_faces = {}

        for path in glob(build_path('faces', '*.jpg')):
            name = name_from_path(path)
            image = io.imread(path)
            try:
                faces = faces_from_image(image)
                face = faces[0] if faces else None
                face_vector = face_to_vector(image, face)
                analyzed_faces[name] = face_vector
            except Exception as e:
                screen.print_at(str(e), 2, 1)

        numpy.save('faces/faces', analyzed_faces)

    utilities.webcam_utility.begin_capture(screen)

if __name__ == '__main__':
    Screen.wrapper(main)
