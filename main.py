import cv2
from sliding_window import SlidingWindow
import glob
import subprocess
import os

#my_image = cv2.imread("/Users/michal/PycharmProjects/HOG_TSR/dataset/image.jpg")
my_image = cv2.imread("/Users/michal/PycharmProjects/HOG_TSR/dataset/test_images/test_5.jpg")
# cv2.imshow("Image", my_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

'''
sliding_window_parameters = {
    'image': my_image,
    'x_win_len': 200,
    'y_win_len': 200,
    'x_increment': 100,
    'y_increment': 100,
    'svc_path': 'trained_models/SVC_2019521.pkl',
    'scaler_path': 'trained_models/scaler_2019521.pkl'
}

sw = SlidingWindow(sliding_window_parameters)


img, a = sw.perform()

print(a)
ratio = img.shape[1] // img.shape[0]
img = cv2.resize(img, (800, 800 * ratio))
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''


def test_on_images():

    sliding_window_parameters = {
        'image': my_image,
        'x_win_len': 200,
        'y_win_len': 200,
        'x_increment': 100,
        'y_increment': 100,
        'svc_path': 'trained_models/SVC_2019521.pkl',
        'scaler_path': 'trained_models/scaler_2019521.pkl'
    }

    sw = SlidingWindow(sliding_window_parameters)

    path_to_test_images_dir = './dataset/test_images/'
    path_to_output_images_dir = path_to_test_images_dir + '/output/'
    test_image_paths = glob.glob(path_to_test_images_dir + '*.jpg')

    if not os.path.exists(path_to_output_images_dir):
        subprocess.run(['mkdir', path_to_output_images_dir])

    for image_path in test_image_paths:
        image = cv2.imread(image_path)

        sw.update_frame(image)
        image, logs = sw.perform()
        name = image_path.split('/')[3]
        cv2.imwrite(path_to_output_images_dir + name, image)

        # dodać zapis logów do pliku tekstowego

test_on_images()