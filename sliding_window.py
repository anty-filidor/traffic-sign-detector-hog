import cv2
from skimage import transform
from termcolor import colored
from object_detector import ObjectDetector
import numpy as np


class SlidingWindow:

    def __init__(self, parameters, image=None):
        """
        This method creates sliding window within SVC performs classification.
        :param parameters: dictionary with parameters:
            image - source image
            x_win_len - length in x axis of sliding window in pixels
            y_win_len - length in y axis of sliding window in pixels
            x_increment - increment in x axis of sliding window in pixels
            y_increment - increment in x axis of sliding window in pixels
            svc_path - path to supported vector classifier
            scaler_path - path to scaler for SVC
        """

        if image is None:
            self.image = None
            self.object_detector = ObjectDetector(parameters['svc_path'], parameters['scaler_path'])
        else:
            self.image = image
            self.object_detector = ObjectDetector(parameters['svc_path'], parameters['scaler_path'], self.image)

        self.x_win_len = parameters['x_win_len']
        self.y_win_len = parameters['y_win_len']
        self.x_increment = parameters['x_increment']
        self.y_increment = parameters['y_increment']

        self.pred_bboxes = []
        self.pred_classes = []
        self.confidences = []

    '''
    def fit(self):
        new_height = self.y_increment
        new_width = self.x_increment
        for i in range(0, self.height):
            new_height += i * self.y_increment
            if new_height + self.y_increment > self.height:
                self.height = new_height
                break

        for i in range(0, self.width):
            new_width += i * self.x_increment
            if new_width + self.x_increment > self.width:
                self.width = new_width
                break

        self.image = cv2.resize(self.image, (self.width, self.height))
    '''

    def image_pyramids(self):
        """
        This method generates image pyramid of self.image.
        :return: layer in pyramid of the images and its shape
        """
        for (i, resized) in enumerate(transform.pyramid_gaussian(self.image, downscale=1.5, multichannel=True)):
            if resized.shape[0] < self.x_win_len or resized.shape[1] < self.y_win_len:
                break
            # print(colored("Layer {}, image dimensions {}, {}".format(i, resized.shape[0], resized.shape[1]), "green"))
            if resized.dtype is not np.int:
                resized = 255 * resized
                resized = resized.astype(np.uint8)
            yield(resized, resized.shape)

    def sliding_window(self):
        """
        This method-generator cuts sliding window from given layer of image pyramid.
        :return: window, cordinates of sliding window in the layer reference, shape of the layer
        """
        for img, shape in self.image_pyramids():
            for y in range(0, img.shape[0], self.y_increment):
                # print("Window coordinates: y: {:3}, x in range 0-{:3} by {:3}".format(y, img.shape[1], self.x_increment))
                for x in range(0, img.shape[1], self.x_increment):
                    yield (img, x, y, img[y:y + self.y_win_len, x:x + self.x_win_len], shape)
                    # CZY TU TRZEBA DAWAĆ CO CHWILĘ TĄ WARSTWĘ????

    def reverse_pyramid(self, layer_shape, x, y):
        """
        This method rescales coordinates of sliding window to coordinates in reference of original layer
            (neither layer).
        :param layer_shape: shape of the layer
        :param x: x coordinate of first vertex of the window
        :param y: y coordinate of first vertex of the window
        :return: tuple of coordinates of two opposing vertexes in the reference of original image
        """
        y_ratio = self.image.shape[0] / layer_shape[0]
        x_ratio = self.image.shape[1] / layer_shape[1]
        vertex_a_x = int(x * x_ratio)
        vertex_a_y = int(y * y_ratio)
        vertex_b_x = int((x + self.x_win_len) * x_ratio)
        vertex_b_y = int((y + self.y_win_len) * y_ratio)

        return vertex_a_x, vertex_a_y, vertex_b_x, vertex_b_y

    def perform(self, mark_on_image=True, track_progress=False):
        """
        This method performs detection
        :param mark_on_image: if true don't decorate given image
        :param track_progress: if true tracks progress of classification
        :return: tuple with image (decorated or not),
        """
        for(img_layer, x, y, window, layer_shape) in self.sliding_window():
            i = 255 * layer_shape[0] / self.image.shape[0]
            detection = self.object_detector.classify(window)
            if detection != 0:
                win_coord = self.reverse_pyramid(layer_shape, x, y)
                if mark_on_image:
                    cv2.rectangle(self.image, (win_coord[0], win_coord[1]), (win_coord[2], win_coord[3]),
                                  (0, i, 255 - i), int(0.01*i)+1)
                self.pred_bboxes.append(win_coord)
                self.pred_classes.append(int(detection))
                self.confidences.append(0.965) # zmienić na prewdziwą confidence

                if track_progress:
                    cv2.rectangle(img_layer, (x, y), (x + self.x_win_len, y + self.y_win_len), (0, 255, 0), 2)
                    cv2.imshow("Window", img_layer)
                    cv2.waitKey(1)

        return self.image, self.pred_bboxes, self.pred_classes, self.confidences

    def update_frame(self, image):
        """
        This method uptades self.image field in sliding window.
        It cleans also: self.pred_bboxes, self.pred_classes, self.confidences.
        :param image: new value of self.image
        """
        self.image = image
        self.pred_bboxes = []
        self.pred_classes = []
        self.confidences = []


'''
my_image = cv2.imread("/Users/michal/PycharmProjects/HOG_TSR/dataset/test_images/test_3.jpg")
#my_image = cv2.imread('/Users/michal/PycharmProjects/HOG_TSR/dataset/image.jpg')

sliding_window_parameters = {
    'x_win_len': 200,
    'y_win_len': 200,
    'x_increment': 100,
    'y_increment': 100,
    'svc_path': 'trained_models/SVC_2019521.pkl',
    'scaler_path': 'trained_models/scaler_2019521.pkl'
}

sw = SlidingWindow(sliding_window_parameters)
sw.update_frame(my_image)
sw.perform(track_progress=True)
my_image = cv2.imread("/Users/michal/PycharmProjects/HOG_TSR/dataset/test_images/test_4.jpg")
sw.update_frame(my_image)
sw.perform(track_progress=True)
'''