import cv2
from skimage import transform
from termcolor import colored
from tsr_object_detector import ObjectDetector
import numpy as np


class SlidingWindow:

    def __init__(self, parameters, image=None, binary_detection=False):

        """
        This method creates sliding window within SVC performs classification.
        :param parameters: dictionary with parameters:
            x_win_len - length in x axis of sliding window in pixels
            y_win_len - length in y axis of sliding window in pixels
            x_increment - increment in x axis of sliding window in pixels
            y_increment - increment in x axis of sliding window in pixels
            svc_path - path to supported vector classifier
            scaler_path - path to scaler for SVC
        :param image: image to be processed
        :param binary_detection: this parameter implicates mode of classification
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

        if binary_detection is True:
            self._label_names = ['no sign', 'sign']
        else:
            self._label_names = ['no sign', 'warning', 'prohibitory', 'mandatory', 'informational']
            #                     0,        1-a,        2-b,            3-c,        4-d

        self._downscale_for_pyramid = parameters['downscale_for_pyramid']

    def _image_pyramids(self):
        """
        This method generates image pyramid of self.image.
        :return: layer in pyramid of the images and its shape
        """
        for (i, resized) in enumerate(transform.pyramid_gaussian(self._image, downscale=self._downscale_for_pyramid,
                                                                 multichannel=True)):
            # if produced layer is smaller than size of window - break
            if resized.shape[0] < self._x_win_len or resized.shape[1] < self._y_win_len:
                break
            # print(colored("Layer {}, image dimensions {}, {}".format(i, resized.shape[0], resized.shape[1]), "green"))
            if resized.dtype is not np.int:
                resized = (255 * resized).astype(np.uint8)
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
                    yield (x, y, img[y:y + self._y_win_len, x:x + self._x_win_len], shape)

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
                    yield (x - self._x_win_len, y - self._y_win_len,
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
        y_ratio = float(self._image.shape[0] / layer_shape[0])
        x_ratio = float(self._image.shape[1] / layer_shape[1])
        vertex_a_x = x * x_ratio
        vertex_a_y = y * y_ratio
        vertex_b_x = (x + self._x_win_len) * x_ratio
        vertex_b_y = (y + self._y_win_len) * y_ratio

        return np.array([vertex_a_x, vertex_a_y, vertex_b_x, vertex_b_y])

    def _non_max_suppression(self, overlapping_threshold):
        """
        This method performs non maximum suppression - it reduces overlapping rois
        :param overlapping_threshold: threshold of overlapping. Above it the roi is deleted
        """

        # initialize the list of picked indexes
        pick = []

        # grab the coordinates of the bounding boxes
        x1 = self._pred_bboxes[:, 0]
        y1 = self._pred_bboxes[:, 1]
        x2 = self._pred_bboxes[:, 2]
        y2 = self._pred_bboxes[:, 3]

        # compute the area of the bounding boxes
        area = (x2 - x1 + 1) * (y2 - y1 + 1)

        # sort the bounding boxes by the vertex b of the bounding box
        indices = np.argsort(y2)

        # keep looping while some indexes still remain in the indices list
        while len(indices) > 0:
            # grab the last index in the indexes list and add the index value to the list of picked indexes
            last = len(indices) - 1
            i = indices[last]
            pick.append(i)

            # find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[indices[:last]])
            yy1 = np.maximum(y1[i], y1[indices[:last]])
            xx2 = np.minimum(x2[i], x2[indices[:last]])
            yy2 = np.minimum(y2[i], y2[indices[:last]])

            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap
            overlap = (w * h) / area[indices[:last]]

            # delete all indexes from the index list that have
            indices = np.delete(indices, np.concatenate(([last], np.where(overlap > overlapping_threshold)[0])))

        # save in _pred_bboxes only rois, which were picked do it also for class labels
        self._pred_bboxes = self._pred_bboxes[pick].astype("int")
        self._pred_classes = [self._pred_classes[i] for i in pick]

    def _mark_on_image(self):
        """
        This method decorates image, by painting RoIs on it.
        """
        self._pred_bboxes = self._pred_bboxes.astype(int)
        for roi, prediction, confidence in zip(self._pred_bboxes, self._pred_classes, self._confidences):
            # prepare text
            text = self._label_names[prediction] + ' {}%'.format(round(confidence * 100, 2))
            # prepare roi
            cv2.rectangle(self._image, (roi[0], roi[1]), (roi[2], roi[3]),
                          (0, 0, confidence * 255), 2)
            # prepare background for label
            size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .7, 1)
            cv2.rectangle(self._image, (roi[0], roi[1]),
                          (roi[0] + size[0][0], roi[1] + size[0][1]), (0, 0, confidence * 255), -1)
            # write a label name
            cv2.putText(self._image, text, (roi[0], roi[1] + size[0][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), lineType=cv2.LINE_AA)

    def perform(self, mark_on_image=True, track_progress=False):
        """
        This method performs detection
        :param mark_on_image: if true don't decorate given image
        :param track_progress: if true tracks progress of classification
        :return: tuple with image (decorated or not),
        """
        for(x, y, window, layer_shape) in self._sliding_window_prec():
            label, confidence = self._object_detector.classify(window)
            if label != 0 and confidence > 0.9:
                win_coord = self._reverse_pyramid(layer_shape, x, y)
                self._pred_bboxes.append(win_coord)
                self._pred_classes.append(int(label))
                self._confidences.append(confidence)

                if track_progress:
                    copy = self._image.copy()
                    cv2.rectangle(copy, (int(win_coord[0]), int(win_coord[1])),
                                  (int(win_coord[2]), int(win_coord[3])), (0, 0, 255), 2)
                    cv2.imshow("Window", copy)
                    cv2.waitKey(0)

        self._pred_bboxes = np.array(self._pred_bboxes)
        if len(self._pred_bboxes > 1):
            self._non_max_suppression(0.01)

        if mark_on_image:
            self._mark_on_image()
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
        for (_, _, window, _) in self._sliding_window_fast():
            yield window


'''
sliding_window_parameters = {
    'x_win_len': 120,
    'y_win_len': 120,
    'x_increment': 90,
    'y_increment': 90,
    'svc_path': 'trained_models/SVC_2019527.pkl',
    'scaler_path': 'trained_models/scaler_2019527.pkl',
    'downscale_for_pyramid': 1.5
}
sw = SlidingWindow(sliding_window_parameters)

my_image = cv2.imread("/Users/michal/PycharmProjects/HOG_TSR/dataset/test_images/test_10.jpg")
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
