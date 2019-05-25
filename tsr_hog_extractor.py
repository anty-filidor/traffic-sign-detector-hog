from skimage.feature import hog
from helpers import convert
import numpy as np
import cv2


class HOGExtractor:

    def __init__(self, parameters, image=None):
        """
        This method creates HOGExtractor object
        :param parameters: given parameters of HOG extractor
        :param image: image to be processed
        """

        self._color_model = parameters['color_model']
        self._ori = parameters['number_of_orientations']
        self._ppc = (parameters['pixels_per_cell'], parameters['pixels_per_cell'])
        self._cpb = (parameters['cells_per_block'], parameters['cells_per_block'])
        self._do_sqrt = parameters['do_transform_sqrt']

        self._ABC_img = None
        self._dims = (None, None, None)
        self._hogA, self._hogB, self._hogC = None, None, None
        self._hogA_img, self._hogB_img, self._hogC_img = None, None, None

        self.RGB_img = None

        self.svc_input_size = parameters['svc_input_size']

        # calculate HOG features if frame is given
        if image is not None:
            self.RGB_img = image
            self._new_frame(self.RGB_img)

    def _hog(self, channel):
        """
        This function calculates HOG of given channel
        :param channel: gray scale image or channel of color image
        :return: HOG feature descriptor array and visualisation of it
        """
        features, hog_img = hog(channel,
                                orientations=self._ori,
                                pixels_per_cell=self._ppc,
                                cells_per_block=self._cpb,
                                transform_sqrt=self._do_sqrt,
                                visualize=True,
                                feature_vector=False,
                                block_norm='L1')
        return features, hog_img

    def _new_frame(self, frame):
        """
        This function calulates HOG and its visualisation of given frame
        :param frame: new image (if passed)
        """
        frame = cv2.resize(frame, (self.svc_input_size, self.svc_input_size))
        self.RGB_img = frame
        self._ABC_img = convert(frame, src_model='rgb', dest_model=self._color_model)
        self._dims = self.RGB_img.shape

        self._hogA, self._hogA_img = self._hog(self._ABC_img[:, :, 0])
        self._hogB, self._hogB_img = self._hog(self._ABC_img[:, :, 1])
        self._hogC, self._hogC_img = self._hog(self._ABC_img[:, :, 2])

    def features(self, frame=None):
        """
        This function gives back HOG features for furhter classification
        :param frame: new image (if passed)
        :return: HOG descriptor of image
        """
        if frame is not None:
            self._new_frame(frame)

        return np.hstack((self._hogA.ravel(), self._hogB.ravel(), self._hogC.ravel()))

    def visualize(self):
        """
        This function gives back calculates HOG for visualisation
        :return: rgb image and visualisation of HOG for each channel
        """
        return self.RGB_img, self._hogA_img, self._hogB_img, self._hogC_img

'''
hog_parameters = {
    'color_model': 'hsv',  # hls, hsv, yuv, ycrcb
    'svc_input_size': 64,  #
    'number_of_orientations': 11,  # 6 - 12
    'pixels_per_cell': 16,  # 8, 16
    'cells_per_block': 2,  # 1, 2
    'do_transform_sqrt': True,
}

my_image = cv2.imread("/Users/michal/PycharmProjects/HOG_TSR/dataset/test_images/test_5.jpg")
he = HOGExtractor(hog_parameters)
print(he.features())
'''
