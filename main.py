import cv2
from sliding_window import SlidingWindow
import glob
import subprocess
import os
from tqdm import tqdm
from termcolor import colored
import pickle

#my_image = cv2.imread("/Users/michal/PycharmProjects/HOG_TSR/dataset/image.jpg")
#my_image = cv2.imread("/Users/michal/PycharmProjects/HOG_TSR/dataset/test_images/test_5.jpg")
# cv2.imshow("Image", my_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def test_on_images():
    print(colored("Starting processing images", 'red'))

    # create sliding window object
    sliding_window_parameters = {
        'x_win_len': 200,
        'y_win_len': 200,
        'x_increment': 100,
        'y_increment': 100,
        'svc_path': 'trained_models/SVC_2019521.pkl',
        'scaler_path': 'trained_models/scaler_2019521.pkl'
    }
    sw = SlidingWindow(sliding_window_parameters)
    print(colored("Created SlidingWindow object", 'red'))

    # create directories to images
    path_to_test_images_dir = './dataset/test_images/'
    path_to_output_images_dir = path_to_test_images_dir + 'output/'
    test_image_paths = glob.glob(path_to_test_images_dir + '*.jpg')

    # if there is no folder "output" create it
    if not os.path.exists(path_to_output_images_dir):
        subprocess.run(['mkdir', path_to_output_images_dir])
        print(colored(("Created directory: " + path_to_output_images_dir), 'red'))

    # cerate list for output logs to further MAP (or IoU) calculation
    detections_output_result = []

    # process all images in folder
    print(colored("Processing:", 'red'))
    for image_path in tqdm(test_image_paths):

        name = image_path.split('/')[3]
        image = cv2.imread(image_path)

        sw.update_frame(image)
        image, pred_bboxes, pred_classes, confidences = sw.perform()

        cv2.imwrite(path_to_output_images_dir + name, image)

        logs = {
            'file': name,  # filename
            'file_path': path_to_test_images_dir,  # path to file

            'gt_classes': [0],  # ground truth classes labels list
            'gt_bboxes': [0],  # list of ground truth bounding boxes

            'pred_classes': pred_classes,  # predicted classes labels list
            'pred_bboxes': pred_bboxes,  # list of predicted bounding boxes (bb are tuples)
            'confidences': confidences,  # list of prediction confidences
        }
        detections_output_result.append(logs)

    # Pickle the 'detections_output_result' list using the highest protocol available.
    with open(path_to_output_images_dir + 'detections_output_result.pkl', 'wb') as f:
        pickle.dump(detections_output_result, f, pickle.HIGHEST_PROTOCOL)


test_on_images()

