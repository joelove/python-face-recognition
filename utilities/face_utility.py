import dlib
import numpy
import os

from glob import glob
from skimage import io

from utilities.path_utility import build_path

FACES_DIRECTORY = 'faces';
FACES_FILENAME = 'faces.npy'

ANALYZED_FACES_PATH = build_path(FACES_DIRECTORY, FACES_FILENAME)

face_detector = dlib.get_frontal_face_detector()
face_recognition = dlib.face_recognition_model_v1('./models/dlib_face_recognition_resnet_model_v1.dat')
shape_predictor = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')

def face_to_vector(image, face):
    face_descriptor = face_recognition.compute_face_descriptor(image, face)
    face_vector = numpy.array(face_descriptor).astype(float)

    return face_vector

def face_from_image(image, face):
    size = face.height() * face.width()
    shape = shape_predictor(image, face)

    return size, shape

def faces_from_image(image):
    detected_faces = face_detector(image, 0)
    faces_in_image = [face_from_image(image, face) for face in detected_faces]
    sorted_faces = [face for _, face in sorted(faces_in_image, reverse=True)]

    return sorted_faces

def analyze_faces():
    if (not os.path.isfile(ANALYZED_FACES_PATH)):
        analyzed_faces = {}

        for path in glob(build_path(FACES_DIRECTORY, '*.jpg')):
            name = name_from_path(path)
            image = io.imread(path)

            try:
                faces = faces_from_image(image)
                face = faces[0] if faces else None
                face_vector = face_to_vector(image, face)
                analyzed_faces[name] = face_vector

            except Exception as e:
                pass

        numpy.save(ANALYZED_FACES_PATH, analyzed_faces)
