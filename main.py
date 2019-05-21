import cv2
from sliding_window import SlidingWindow

my_image = cv2.imread("/Users/michal/PycharmProjects/HOG_TSR/dataset/image.jpg")
# cv2.imshow("Image", my_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

sliding_window_parameters = {
    'image': my_image,
    'x_win_len': 200,
    'y_win_len': 200,
    'x_increment': 100,
    'y_increment': 100,
    'svc_path': 'trained_models/SVC_201954.pkl',
    'scaler_path': 'trained_models/scaler_201954.pkl'
}
sw = SlidingWindow(sliding_window_parameters)
_, a = sw.perform()

print(len(a))

