from skimage.feature import hog
from helpers import convert
import numpy as np
import cv2


class HOGExtractor:

    def __init__(self, parameters, image):
        """
        This method creates HOGExtractor object
        :param parameters: given parameters of HOG extractor
        :param image: image to be processed
        """

        self.color_model = parameters['color_model']
        self.ori = parameters['number_of_orientations']
        self.ppc = (parameters['pixels_per_cell'], parameters['pixels_per_cell'])
        self.cpb = (parameters['cells_per_block'], parameters['cells_per_block'])
        self.do_sqrt = parameters['do_transform_sqrt']

        self.ABC_img = None
        self.dims = (None, None, None)
        self.hogA, self.hogB, self.hogC = None, None, None
        self.hogA_img, self.hogB_img, self.hogC_img = None, None, None

        self.RGB_img = image

        self.svc_input_size = parameters['svc_input_size']

        # calculate HOG features for given frame
        self.new_frame(self.RGB_img)

    def hog(self, channel):
        """
        This function calculates HOG of given channel
        :param channel: gray scale image or channel of color image
        :return: HOG feature descriptor array and visualisation of it
        """
        features, hog_img = hog(channel,
                                orientations=self.ori,
                                pixels_per_cell=self.ppc,
                                cells_per_block=self.cpb,
                                transform_sqrt=self.do_sqrt,
                                visualize=True,
                                feature_vector=False,
                                block_norm='L1')
        return features, hog_img

    def new_frame(self, frame):
        """
        This function calulates HOG and its visualisation of given frame
        :param frame: new image (if passed)
        """
        frame = cv2.resize(frame, (self.svc_input_size, self.svc_input_size))
        self.RGB_img = frame
        self.ABC_img = convert(frame, src_model='rgb', dest_model=self.color_model)
        self.dims = self.RGB_img.shape

        self.hogA, self.hogA_img = self.hog(self.ABC_img[:, :, 0])
        self.hogB, self.hogB_img = self.hog(self.ABC_img[:, :, 1])
        self.hogC, self.hogC_img = self.hog(self.ABC_img[:, :, 2])

    def features(self, frame=None):
        """
        This function gives back HOG features for furhter classification
        :param frame: new image (if passed)
        :return: HOG descriptor of image
        """
        if frame is not None:
            self.new_frame(frame)

        return np.hstack((self.hogA.ravel(), self.hogB.ravel(), self.hogC.ravel()))

    def visualize(self):
        """
        This function gives back calculates HOG for visualisation
        :return: rgb image and visualisation of HOG for each channel
        """
        return self.RGB_img, self.hogA_img, self.hogB_img, self.hogC_img

