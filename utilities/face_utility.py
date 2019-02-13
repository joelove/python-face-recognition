import dlib
import numpy


face_detector = dlib.get_frontal_face_detector()
face_recognition = dlib.face_recognition_model_v1('./models/dlib_face_recognition_resnet_model_v1.dat')
shape_predictor = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')


def face_to_vector(image, face):
    return (
        numpy
            .array(face_recognition.compute_face_descriptor(image, face))
            .astype(float)
    )

def faces_from_image(image):
    upsampling_factor = 0
    detected_faces = face_detector(image, upsampling_factor)

    faces = [
        (face.height() * face.width(), shape_predictor(image, face))
        for face in detected_faces
    ]

    return [face for _, face in sorted(faces, reverse=True)]

def identify_face(image):
    faces = faces_from_image(image)
    analyzed_faces = numpy.load('faces/faces.npy').item()
    face_identifiers = numpy.array(list(analyzed_faces.keys()))
    face_matrix = numpy.array(list(analyzed_faces.values()))

    if (faces):
        descriptor = face_recognition.compute_face_descriptor(image, faces[0])
        face_vector = numpy.array(descriptor).astype(float)
        differences = numpy.subtract(numpy.array(face_matrix), face_vector)
        distances = numpy.linalg.norm(differences, axis=1)
        closest_index = numpy.argmin(distances)

        return face_identifiers[closest_index], distances[closest_index]

    return None, 0

def load_faces():
    numpy.load('faces.npy').item()
