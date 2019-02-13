import dlib
import numpy


face_detector = dlib.get_frontal_face_detector()
face_recognition = dlib.face_recognition_model_v1('./models/dlib_face_recognition_resnet_model_v1.dat')
shape_predictor = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')

analyzed_faces = {}
face_identifiers = []
face_matrix = []


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

def identity_face(image, face):
    global analyzed_faces, face_identifiers, face_matrix

    face_vector = face_to_vector(image, face)
    differences = numpy.subtract(face_matrix, face_vector)
    distances = numpy.linalg.norm(differences, axis=1)
    closest_index = numpy.argmin(distances)
    face_identifier = face_identifiers[closest_index]
    distance = distances[closest_index]

    return face_identifier, distance

def identify_faces(image):
    global analyzed_faces, face_identifiers, face_matrix

    analyzed_faces = numpy.load('faces/faces.npy').item()
    face_identifiers = numpy.array(list(analyzed_faces.keys()))
    face_matrix = numpy.array(list(analyzed_faces.values()))
    faces_in_image = faces_from_image(image)
    identified_faces = [identity_face(image, face) for face in faces_in_image]

    return identified_faces
