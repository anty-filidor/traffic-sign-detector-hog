from tsr_svc import SVC
from tsr_hog_extractor import HOGExtractor
import cv2
import tsr_helpers
import warnings
from sklearn.externals import joblib
from tsr_helpers import get_hog_parameters


class ObjectDetector:

    def __init__(self, svc_path=None, scaler_path=None, image=None, hog_parameters=None):
        """
        This method initialise object of ObjectDetector class
        :param image: image to be processed, exactly the slice of image cut by sliding window
        :param svc_path: path to SVC
        :param scaler_path: path to scaler for SVC
        :param hog_parameters: parameters of HOG extractor
        """
        if svc_path is None or scaler_path is None:
            warnings.warn("SVC path or scaler path not passed! Used default classifier.")
            self.classifier = SVC(joblib.load('trained_models/svc.pkl'), joblib.load('trained_models/scaler.pkl'))
        else:
            self.classifier = SVC(joblib.load(svc_path), joblib.load(scaler_path))

        if hog_parameters is None:
            self._hog_parameters = get_hog_parameters()
        else:
            self._hog_parameters = hog_parameters

        if image is None:
            self.image = None
            self.HOG = HOGExtractor(self._hog_parameters)
        else:
            self.image = image
            self.HOG = HOGExtractor(self._hog_parameters, self.image)

    def demonstration(self, image=None):
        """
        This method demonstrates HOG features and performs classification of the image
        :param image: optional parameter - image, if not method works on self.image
        """
        # if new image passed change self.image and calculate new features for it
        if image is not None:
            self.image = image
            f = self.HOG.features(self.image)
        else:
            try:
                f = self.HOG.features()
            except ValueError:
                print("Object: ", self.__class__, " don't have image passed! Provide image to them,"
                                                  " then perform classification.")
        print("feature shape:", f.shape)
        rgb_img, a_img, b_img, c_img = self.HOG.visualize()
        tsr_helpers.show_images([rgb_img, a_img, b_img, c_img], per_row=4, per_col=1, W=20, H=4)
        print(self.classifier.predict(f))

    def classify(self, image=None):
        """
        This method performs only classification on the image
        :param image: optional parameter - image, if not method works on self.image
        :return: prediction - 1 is true, 0 is false
        """
        if image is not None:
            self.image = image
            f = self.HOG.features(self.image)
        else:
            try:
                f = self.HOG.features()
            except ValueError:
                print("Object: ", self.__class__, " don't have image passed! Provide image to them,"
                                                  " then perform classification.")
        return self.classifier.predict(f)


'''
my_image_path = '/Users/michal/PycharmProjects/HOG_TSR/dataset/image.jpg'
my_image = cv2.imread(my_image_path)

for i in range(0, my_image.shape[0]):
    for j in range(0, my_image.shape[1]):
        print(my_image[i][j][:])

cv2.imshow("Image", my_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

svc_path = "trained_models/svc.pkl"
scaler_path = "trained_models/scaler.pkl"

od = ObjectDetector(svc_path, scaler_path, my_image)
od.demonstration()
# od.demonstration(cv2.imread('./dataset/test_images/test_10.png'))
print(od.classify())
'''
