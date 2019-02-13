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

    return numpy.array(face_descriptor).astype(float)

def face_from_image(image, face):
    size = face.height() * face.width()
    shape = shape_predictor(image, face)

    return size, shape

def faces_from_image(image):
    upsampling_factor = 0
    detected_faces = face_detector(image, upsampling_factor)
    faces = [face_from_image(image, face) for face in detected_faces]

    return [face for _, face in sorted(faces, reverse=True)]

def identity_face(image, face):
    global analyzed_faces, face_identifiers, face_matrix

    face_vector = face_to_vector(image, face)
    differences = numpy.subtract(face_matrix, face_vector)
    distances = numpy.linalg.norm(differences, axis=1)
    closest_index = numpy.argmin(distances)

    return face_identifiers[closest_index], distances[closest_index]

def identify_faces(image):
    global analyzed_faces, face_identifiers, face_matrix

    analyzed_faces = numpy.load('faces/faces.npy').item()
    face_identifiers = numpy.array(list(analyzed_faces.keys()))
    face_matrix = numpy.array(list(analyzed_faces.values()))

    return [identity_face(image, face) for face in faces_from_image(image)]

def load_faces():
    numpy.load('faces.npy').item()
