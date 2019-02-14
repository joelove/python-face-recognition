import dlib
import numpy
import os

from glob import glob
from skimage import io
from funcy import compose, first


FACES_DIRECTORY = 'faces';
FACES_FILENAME = 'faces.npy'

MODELS_DIRECTORY = 'models';
MODELS_FACE_RECOGNITION_FILENAME = 'dlib_face_recognition_resnet_model_v1.dat';
MODELS_SHAPE_PREDICTOR_FILENAME = 'shape_predictor_68_face_landmarks.dat';

FACES_FILE_PATH = os.path.join(FACES_DIRECTORY, FACES_FILENAME)
FACE_RECOGNITION_FILE_PATH = os.path.join(MODELS_DIRECTORY, MODELS_FACE_RECOGNITION_FILENAME)
SHAPE_PREDICTOR_FILE_PATH = os.path.join(MODELS_DIRECTORY, MODELS_SHAPE_PREDICTOR_FILENAME)


get_file_name = compose(first, os.path.splitext, os.path.basename)
glob_join = compose(glob, os.path.join)
array_from_list = compose(numpy.array, list)
face_detector = dlib.get_frontal_face_detector()
face_recognition = dlib.face_recognition_model_v1(FACE_RECOGNITION_FILE_PATH)
shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_FILE_PATH)


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


def identify_faces(image):
    analyzed_faces = numpy.load(FACES_FILE_PATH).item()
    face_identifiers = array_from_list(analyzed_faces.keys())
    face_matrix = array_from_list(analyzed_faces.values())

    def identity_face(face):
        face_vector = face_to_vector(image, face)
        differences = numpy.subtract(face_matrix, face_vector)
        distances = numpy.linalg.norm(differences, axis=1)
        closest_index = numpy.argmin(distances)
        face_identifier = face_identifiers[closest_index]
        distance = distances[closest_index]

        return face_identifier, distance

    faces_in_image = faces_from_image(image)
    identified_faces = map(identity_face, faces_in_image)

    return identified_faces


def create_faces_file():
    if (os.path.isfile(FACES_FILE_PATH)):
        return

    analyzed_faces = {}

    for path in glob_join(FACES_DIRECTORY, '*.jpg'):
        image = io.imread(path)
        faces = faces_from_image(image)

        if (not faces):
            break

        face = first(faces)
        name = get_file_name(path)
        face_vector = face_to_vector(image, face)
        analyzed_faces[name] = face_vector

    numpy.save(FACES_FILE_PATH, analyzed_faces)
