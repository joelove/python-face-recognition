import numpy

from glob import glob
from skimage import io

from utilities.face_utility import face_to_vector, faces_from_image
from utilities.path_utility import build_path, name_from_path


class FaceAnalyser:
    enrolled_faces = {}

    def enroll_face(image, name):
      try:
        faces = faces_from_image(image)
        face = faces[0] if faces else None
        face_vector = face_to_vector(image, face)
        enrolled_faces[name] = face_vector
      except Exception as e:
        print('Unable to enroll', name)
        print(e)

    def enroll_faces():
        glob_pattern = build_path('faces', '*.jpg')
        for path in glob(glob_pattern):
          name = name_from_path(path)
          image = io.imread(path)
          enroll_face(image, name)

    def save_faces():
        numpy.save('faces.npy', enrolled_faces)

    def load_faces():
        enrolled_faces = numpy.load('faces.npy').item()
