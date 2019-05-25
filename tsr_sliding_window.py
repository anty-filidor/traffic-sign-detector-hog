import cv2
from skimage import transform
from termcolor import colored
from tsr_object_detector import ObjectDetector
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
            self._image = None
            self._object_detector = ObjectDetector(parameters['svc_path'], parameters['scaler_path'])
        else:
            self._image = image
            self._object_detector = ObjectDetector(parameters['svc_path'], parameters['scaler_path'], self._image)

        self._x_win_len = parameters['x_win_len']
        self._y_win_len = parameters['y_win_len']
        self._x_increment = parameters['x_increment']
        self._y_increment = parameters['y_increment']

        self._pred_bboxes = []
        self._pred_classes = []
        self._confidences = []

        self._label_names = ['sign']

        # DODAĆ WYCZTYWANIE LABELI KLAS!£!!!

    def _image_pyramids(self):
        """
        This method generates image pyramid of self.image.
        :return: layer in pyramid of the images and its shape
        """
        for (i, resized) in enumerate(transform.pyramid_gaussian(self._image, downscale=1.5, multichannel=True)):
            # if produced layer is smaller than size of window - break
            if resized.shape[0] < self._x_win_len or resized.shape[1] < self._y_win_len:
                break
            # print(colored("Layer {}, image dimensions {}, {}".format(i, resized.shape[0], resized.shape[1]), "green"))
            if resized.dtype is not np.int:
                resized = 255 * resized
                resized = resized.astype(np.uint8)
            yield(resized, resized.shape)

    def _sliding_window_fast(self):
        """
        This method-generator cuts sliding window from given layer of image pyramid. It is fast, because we do not chan-
        ge aa incrementation value, but in consequence we do not reach all boundaries of image.
        :return: window, cordinates of sliding window in the layer reference, shape of the layer
        """
        # Iterate through all layers from image pyramid
        for img, shape in self._image_pyramids():
            # Iterate in y axis through image. y and x are left-up coordinates of window. We do iteration from left
            # to right, up to bottom.
            for y in range(0, img.shape[0] - self._y_win_len, self._y_increment):
                # print("Window coord: y: {:3}, x in range 0-{:3} by {:3}".format(y, img.shape[1], self.x_increment))
                for x in range(0, img.shape[1] - self._x_win_len, self._x_increment):
                    yield (img, x, y, img[y:y + self._y_win_len, x:x + self._x_win_len], shape)
                    # CZY TU TRZEBA DAWAĆ CO CHWILĘ TĄ WARSTWĘ????

    def _sliding_window_prec(self):
        """
        This method-generator cuts sliding window from given layer of image pyramid. It is precise, because it perform a
        scaling of incrementation value to reach all boundaries of image. By that it is slower.
        :return: window, cordinates of sliding window in the layer reference, shape of the layer
        """
        # Iterate through all layers from image pyramid
        for img, shape in self._image_pyramids():

            # Usually with given incrementation of window in each axis we cannot reach all boundaries of image. To pre-
            # vent it, we calculate a coefficient which chagne an incrementation value to slice all fragment of image

            # Incrementation coefficient for y axis. Notice, that if it is less than 0.2 (apriori value), we treat a
            # size in y axis of image as big enough to extract only one window in this axis.
            incr_y_coef = (img.shape[0] - self._y_win_len) / self._y_increment
            if incr_y_coef > 1:
                incr_y_coef = incr_y_coef / int(incr_y_coef)
            elif incr_y_coef < 0.2:
                incr_y_coef = img.shape[0]

            # Incrementation coefficient for x axis. Notice, that if it is less than 0.2 (apriori value), we treat a
            # size in x axis of image as big enough to extract only one window in this axis.
            incr_x_coef = (img.shape[1] - self._x_win_len) / self._x_increment
            if incr_x_coef > 1:
                incr_x_coef = incr_x_coef / int(incr_x_coef)
            elif incr_x_coef < 0.2:
                incr_x_coef = img.shape[1]

            # Iterate in y axis through image. y and x are right-down coordinates of window. We do iteration from left
            # to right, up to bottom.
            y = self._y_win_len
            while y <= img.shape[0]:
                # print("Window coord: y: {:3}, x in range 0-{:3} by {:3}".format(y, img.shape[1], self.x_increment))
                x = self._x_win_len
                while x <= img.shape[1]:
                    yield (img, x - self._x_win_len, y - self._y_win_len,
                           img[y - self._y_win_len:y, x - self._x_win_len:x], shape)
                    x += int(incr_x_coef * self._x_increment)
                y += int(incr_y_coef * self._y_increment)

    def _reverse_pyramid(self, layer_shape, x, y):
        """
        This method rescales coordinates of sliding window to coordinates in reference of original layer
            (neither layer).
        :param layer_shape: shape of the layer
        :param x: x coordinate of first vertex of the window
        :param y: y coordinate of first vertex of the window
        :return: tuple of coordinates of two opposing vertexes in the reference of original image
        """
        y_ratio = self._image.shape[0] / layer_shape[0]
        x_ratio = self._image.shape[1] / layer_shape[1]
        vertex_a_x = int(x * x_ratio)
        vertex_a_y = int(y * y_ratio)
        vertex_b_x = int((x + self._x_win_len) * x_ratio)
        vertex_b_y = int((y + self._y_win_len) * y_ratio)

        return vertex_a_x, vertex_a_y, vertex_b_x, vertex_b_y

    def perform(self, mark_on_image=True, track_progress=False):
        """
        This method performs detection
        :param mark_on_image: if true don't decorate given image
        :param track_progress: if true tracks progress of classification
        :return: tuple with image (decorated or not),
        """
        for(img_layer, x, y, window, layer_shape) in self._sliding_window_prec():
            i = 255 * layer_shape[0] / self._image.shape[0]
            label, confidence = self._object_detector.classify(window)
            if label != 0:
                win_coord = self._reverse_pyramid(layer_shape, x, y)
                if mark_on_image:
                    # mark ROI
                    text = self._label_names[0] + ' {}%'.format(round(confidence*100, 2))
                    cv2.rectangle(self._image, (win_coord[0], win_coord[1]), (win_coord[2], win_coord[3]),
                                  (0, i, 255 - i), int(0.01*i) + 1)
                    # prepare background for label
                    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .7, 1)
                    cv2.rectangle(self._image, (win_coord[0], win_coord[1]),
                                  (win_coord[0] + size[0][0], win_coord[1] + size[0][1]), (0, i, 255 - i), -1)
                    # write a label name
                    cv2.putText(self._image, text, (win_coord[0], win_coord[1] + size[0][1]),
                                cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), lineType=cv2.LINE_AA)
                self._pred_bboxes.append(win_coord)
                self._pred_classes.append(int(label))
                self._confidences.append(confidence)

            if track_progress:
                cv2.rectangle(img_layer, (x, y), (x + self._x_win_len, y + self._y_win_len), (0, 255, 0), 2)
                cv2.imshow("Window", img_layer)
                cv2.waitKey(0)

        return self._image, self._pred_bboxes, self._pred_classes, self._confidences

    def update_frame(self, image):
        """
        This method uptades self.image field in sliding window.
        It cleans also: self.pred_bboxes, self.pred_classes, self.confidences.
        :param image: new value of self.image
        """
        self._image = image
        self._pred_bboxes = []
        self._pred_classes = []
        self._confidences = []

    def generate_slices(self):
        """
        This method generates only slices of the image. It is used for training a classifier
        :return: slice of given image
        """
        for (_, _, _, window, _) in self._sliding_window_fast():
            yield window


'''
sliding_window_parameters = {
    'x_win_len': 120,
    'y_win_len': 120,
    'x_increment': 90,
    'y_increment': 90,
    'svc_path': 'trained_models/SVC_2019521.pkl',
    'scaler_path': 'trained_models/scaler_2019521.pkl'
}
sw = SlidingWindow(sliding_window_parameters)

my_image = cv2.imread("/Users/michal/PycharmProjects/HOG_TSR/dataset/test_images/test_3.jpg")
sw.update_frame(my_image)
my_image, _, _, _ = sw.perform(track_progress=False)
cv2.imshow("Processed", my_image)
cv2.waitKey(0)

my_image = cv2.imread("/Users/michal/PycharmProjects/HOG_TSR/dataset/test_images/test_4.jpg")
sw.update_frame(my_image)
my_image, _, _, _ = sw.perform(track_progress=True)
cv2.imshow("Processed", my_image)
cv2.waitKey(0)
'''
