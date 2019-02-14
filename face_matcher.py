import numpy
import time

from glob import glob

from utilities.face_utility import face_to_vector, faces_from_image

face_identifiers = []
face_matrix = []

def identity_face(image, face):
    face_vector = face_to_vector(image, face)
    print(face_matrix, face_vector)
    time.sleep(10)
    differences = numpy.subtract(face_matrix, face_vector)
    distances = numpy.linalg.norm(differences, axis=1)
    closest_index = numpy.argmin(distances)
    face_identifier = face_identifiers[closest_index]
    distance = distances[closest_index]

    return face_identifier, distance

def identify_faces(image):
    analyzed_faces = numpy.load('faces/faces.npy').item()
    face_identifiers = numpy.array(list(analyzed_faces.keys()))
    face_matrix = numpy.array(list(analyzed_faces.values()))
    faces_in_image = faces_from_image(image)
    identified_faces = [identity_face(image, face) for face in faces_in_image]

    return identified_faces
