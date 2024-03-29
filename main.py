import cv2
from tsr_sliding_window import SlidingWindow
import glob
import subprocess
import os
from tqdm import tqdm
from termcolor import colored
import pickle
import pandas as pd


def detect_on_images(save_images=True, with_bboxes=True):
    """
    This method performs tests HOG+SVM detector on the images in given folder it also prepares data
     to calculate mean average precision
    :param save_images: if true method saves images with marked rois
    :param with_bboxes: if true method add to path_to_output_images_dir.pkl ground truth rois of procesed image
    (of course a file with gt rois is obligatory )
    """
    print(colored("Starting processing images\n", 'red'))

    # create sliding window object
    sliding_window_parameters = {
        'x_win_len': 50,
        'y_win_len': 50,
        'x_increment': 40,
        'y_increment': 40,
        'svc_path': 'trained_models/SVC_2019527.pkl',
        'scaler_path': 'trained_models/scaler_2019527.pkl',
        'downscale_for_pyramid': 1.3
    }
    sw = SlidingWindow(sliding_window_parameters, binary_detection=True)
    print(colored("Created SlidingWindow object\n", 'red'))

    # create directories to images
    path_to_test_images_dir = './dataset/test_images/'
    path_to_output_images_dir = path_to_test_images_dir + 'output/'
    path_to_test_images_dir = '/Users/michal/Tensorflow/datasets/GTSDB/'
    test_image_paths = glob.glob(path_to_test_images_dir + '*.ppm')

    # read csv with ground true bounding boxes
    if with_bboxes:
        path_to_gt_txt = '/Users/michal/Tensorflow/datasets/GTSDB/gt.txt'
        gt = pd.read_csv(path_to_gt_txt, delimiter=';', header=None, names=['file', 'x1', 'y1', 'x2', 'y2', "class_id"])

        # if binary classification uncomment line below
        gt['class_id'] = 1

    # if there is no folder "output" create it
    if not os.path.exists(path_to_output_images_dir):
        subprocess.run(['mkdir', path_to_output_images_dir])
        print(colored(("Created directory: {}\n".format(path_to_output_images_dir)), 'red'))

    # cerate list for output logs to further MAP (or IoU) calculation
    detections_output_result = []

    # process all images in folder
    print(colored("Processing:\n", 'red'))
    for image_path in tqdm((test_image_paths[:200])):

        # name = image_path.split('/')[3]
        name = image_path.split('/')[6]
        image = cv2.imread(image_path)

        # search for selected image's rows in the csv and extract rois and id of classes
        if with_bboxes:
            bb = (gt.loc[gt['file'] == name])
            classes = bb['class_id'].values.tolist()
            gt_bboxes = [tuple(l) for l in bb[['x1', 'y1', 'x2', 'y2']].values]
        else:
            classes = [0]
            gt_bboxes = [0]

        # perform detection
        sw.update_frame(image)
        image, pred_bboxes, pred_classes, confidences = sw.perform(mark_on_image=save_images)
        pred_bboxes = [tuple(l) for l in pred_bboxes]

        if save_images is True:
            cv2.imwrite(path_to_output_images_dir + name, image)

        logs = {
            'file': name,  # filename
            'file_path': image_path,  # path to file

            'gt_classes': classes,  # ground truth classes labels list
            'gt_bboxes': gt_bboxes,  # list of ground truth bounding boxes

            'pred_classes': pred_classes,  # predicted classes labels list
            'pred_bboxes': pred_bboxes,  # list of predicted bounding boxes (bb are tuples)
            'confidences': confidences,  # list of prediction confidences
        }
        detections_output_result.append(logs)

    # Pickle the 'detections_output_result' list using the highest protocol available.
    with open(path_to_output_images_dir + 'detections_output_result.pkl', 'wb') as f:
        pickle.dump(detections_output_result, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    with open(path_to_output_images_dir + 'sliding_windows_params.txt', 'w') as f:
        f.write(str(sliding_window_parameters))
        f.write(str(sw._object_detector._hog_parameters))
        f.close()


detect_on_images(save_images=True, with_bboxes=True)
