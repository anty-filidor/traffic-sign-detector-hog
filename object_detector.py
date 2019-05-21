from svc import SVC
from hog_extractor import HOGExtractor
import cv2
import helpers
from sklearn.externals import joblib


class ObjectDetector:

    def __init__(self, image, svc_path, scaler_path, hog_parameters=None):

        if svc_path is None or scaler_path is None:
            self.classifier = None  # zmienić to póżniej!!!!!!!!!!
        self.classifier = SVC(joblib.load(svc_path), joblib.load(scaler_path))

        self.image = image

        if hog_parameters is None:
            hog_parameters = {
                'color_model': 'hsv',  # hls, hsv, yuv, ycrcb
                'svc_input_size': 64,  #
                'number_of_orientations': 11,  # 6 - 12
                'pixels_per_cell': 16,  # 8, 16
                'cells_per_block': 2,  # 1, 2
                'do_transform_sqrt': True,
            }
        self.HOG = HOGExtractor(hog_parameters, self.image)

    def demonstration(self, image=None):
        if image is not None:
            self.image = image
            f = self.HOG.features(self.image)
        else:
            f = self.HOG.features()
        print("feature shape:", f.shape)
        rgb_img, a_img, b_img, c_img = self.HOG.visualize()
        helpers.show_images([rgb_img, a_img, b_img, c_img], per_row=4, per_col=1, W=20, H=4)
        print(self.classifier.predict(f))

    def classify(self, image=None):
        if image is not None:
            self.image = image
            f = self.HOG.features(self.image)
        else:
            f = self.HOG.features()
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

svc_path = "trained_models/SVC_201954.pkl"
scaler_path = "trained_models/scaler_201954.pkl"

od = ObjectDetector(my_image, svc_path, scaler_path)
# od.demonstration()
# od.demonstration(cv2.imread('./dataset/test_images/test_10.png'))
print(od.classify())
'''

