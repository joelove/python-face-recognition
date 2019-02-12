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
    UPSAMPLING_FACTOR = 0
    faces = [
        (face.height() * face.width(), shape_predictor(image, face))
        for face in face_detector(image, UPSAMPLING_FACTOR)
    ]
    return [face for _, face in sorted(faces, reverse=True)]

def identify_face(image):
    faces = faces_from_image(image)
    face = faces[0] if faces else None

    descriptor = face_recognition.compute_face_descriptor(image, face)
    face_vector = numpy.array(descriptor).astype(float)

    analyzed_faces = np.load('../faces/.faces').item()
    enroll_identifiers = numpy.array(list(analyzed_faces.keys()))
    enroll_matrix = numpy.array(list(analyzed_faces.values()))

    differences = numpy.subtract(numpy.array(enroll_matrix), face_vector)
    distances = numpy.linalg.norm(differences, axis=1)
    closest_index = numpy.argmin(distances)

    return enroll_identifiers[closest_index], distances[closest_index]

def load_faces():
    numpy.load('faces.npy').item()
