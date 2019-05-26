import cv2
import pandas as pd
from tsr_sliding_window import SlidingWindow
from tqdm import tqdm
import glob
import numpy as np
from tsr_hog_extractor import HOGExtractor
from tsr_helpers import get_hog_parameters


class NegativeDataGenerator:

    def __init__(self, dataset_path=None, manually_mode=False):
        """
        This "constructor" just sets a path to originally downloaded GTSDB folder.
        You can get it here: http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset
        :param dataset_path: a path (str)
        :param manually_mode: switch to select slices manually
        """
        if dataset_path is not None:
            self._dataset_path = dataset_path
        else:
            self._dataset_path = '/Users/michal/Tensorflow/datasets/GTSDB/'
        self._manually_mode = manually_mode

    def __call__(self):
        """
        This method is a call for generator of slices from images in GTSDB folder
        :return: a list of slices of the images
        """

        # read file with GT bounding boxes
        # header: ImgNo#.ppm;#leftCol#;##topRow#;#rightCol#;#bottomRow#;#ClassID#
        gt_images = pd.read_csv((self._dataset_path + 'gt.txt'), delimiter=';', header=None,
                                names=['filename'], usecols=[0])
        gt_images = gt_images.drop_duplicates()
        gt_images = gt_images.sort_values(by=['filename'])

        # read all images from the folder
        all_images = []
        images = glob.glob(self._dataset_path + '*.ppm')
        for image_path in images:
            all_images.append(image_path.split('/')[6])
        all_images = pd.DataFrame(all_images, columns=['filename'])
        all_images = all_images.sort_values(by=['filename'])

        # get images which don't have bounding boxes -> they are negative, and also delete some ambiguous images
        negative_images = pd.concat([all_images, gt_images]).drop_duplicates(keep=False)
        negative_images = negative_images.sort_values(by=['filename'])
        negative_images = negative_images[negative_images.filename != '00145.ppm']
        negative_images = negative_images[negative_images.filename != '00235.ppm']
        negative_images = negative_images[negative_images.filename != '00543.ppm']
        negative_images = negative_images[negative_images.filename != '00563.ppm']

        print("Negative images\n", negative_images.describe())
        print("We will slice {} negative images!".format(len(negative_images.index)))
        print("It can take a while ...")

        # create sliding window object
        sliding_window_parameters = {
            'x_win_len': 40,
            'y_win_len': 40,
            'x_increment': 40,
            'y_increment': 40,
            'svc_path': None,
            'scaler_path': None,
            'downscale_for_pyramid': 1.1
        }
        sw = SlidingWindow(sliding_window_parameters)

        if self._manually_mode:
            print("Press:\n\t'e' to select manually slices of image, \n\t'a' to select all slices from image"
                  "\n\t'p' to break importing images or slicing")

        # Iterate through all images and for each one run sliding window to get slices. Then append it to slices list.
        slices = []
        for _, image in tqdm(negative_images.iterrows()):
            img = cv2.imread(self._dataset_path + image['filename'])
            sw.update_frame(img)

            if self._manually_mode:
                cv2.imshow("image", img)
                callback = cv2.waitKey(0)
                if callback == 112:  # p
                    break
                elif callback == 101:  # e - select manually slices
                    for slice in sw.generate_slices():
                        cv2.imshow("slice", slice)
                        callback = cv2.waitKey(0)
                        if callback == 112:  # p
                            break
                        elif callback == 101:  # e
                            slices.append(slice)
                elif callback == 97:  # a - select all slices from image
                    for slice in sw.generate_slices():
                        slices.append(slice)
            else:
                for slice in sw.generate_slices():
                    slices.append(slice)

        print("Extracted {} slices from negative images.".format(len(slices)))
        return slices


ndg = NegativeDataGenerator()
negative_images = ndg()

negative_images = np.asarray(negative_images)
np.save('dataset/extracted_features/negative_images_improved', negative_images)
# negative_images = np.load('trained_models/negative_images_improved.npy')
np.random.shuffle(negative_images)

extractor = HOGExtractor(get_hog_parameters())

negative_features = []
print("Extracting features from non traffic signs...")
for img in tqdm(negative_images[:50000]):
    negative_features.append(extractor.features(img))


negative_features = np.asarray(negative_features)
np.save('dataset/extracted_features/negative_features_improved', negative_features)

