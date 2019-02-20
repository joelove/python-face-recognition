import numpy
import utilities.face_utility

from PIL import Image
from webcam import capture


ret = True
frame = None
faces = []
image = numpy.array(Image.new('L', (100, 100), 0))


def test_capture(mocker):

    video_capture = mocker.stub()
    video_capture.grab = mocker.stub()
    video_capture.retrieve = mocker.MagicMock(return_value=(ret, frame))

    mock_cv2_VideoCapture = mocker.patch('cv2.VideoCapture', autospec=True)
    mock_cv2_VideoCapture.return_value = video_capture

    mock_cv2_resize = mocker.patch('cv2.resize', autoSpec=True)
    mock_cv2_resize.return_value = image

    screen = mocker.stub()
    screen.clear = mocker.stub()
    screen.print_at = mocker.stub()
    screen.refresh = mocker.stub()

    mock_identify_faces = mocker.patch.object(utilities.face_utility, 'identify_faces', autoSpec=True)

    capture(screen)

    mock_cv2_VideoCapture.assert_called_with(0)
    mock_cv2_resize.assert_called_with(frame, None)
    mock_identify_faces.assert_called_with(image)
