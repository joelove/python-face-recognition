import dlib
import numpy
import os

from glob import glob
from skimage import io

FACES_DIRECTORY = 'faces';
FACES_FILENAME = 'faces.npy'

FACES_FILE_PATH = os.path.join(FACES_DIRECTORY, FACES_FILENAME)

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


def identify_faces(image):
    analyzed_faces = numpy.load(FACES_FILE_PATH).item()
    face_identifiers = numpy.array(list(analyzed_faces.keys()))
    face_matrix = numpy.array(list(analyzed_faces.values()))

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

    for path in glob(os.path.join(FACES_DIRECTORY, '*.jpg')):
        image = io.imread(path)
        faces = faces_from_image(image)

        if (not faces):
            break

        name = os.path.splitext(os.path.basename(path))[0]
        face_vector = face_to_vector(image, faces[0])
        analyzed_faces[name] = face_vector

    numpy.save(FACES_FILE_PATH, analyzed_faces)
