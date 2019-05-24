import cv2
import pandas as pd
from sliding_window import SlidingWindow
from tqdm import tqdm
import glob


class NegativeDataGenerator:

    def __init__(self, dataset_path=None):
        """
        This "constructor" just sets a path to originally downloaded GTSDB folder.
        You can get it here: http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset
        :param dataset_path: a path (str)
        """
        if dataset_path is not None:
            self.dataset_path = dataset_path
        else:
            self.dataset_path = '/Users/michal/Tensorflow/datasets/GTSDB/'

    def __call__(self):
        """
        This method is a call for generator of slices from images in GTSDB folder
        :return: a list of slices of the images
        """

        # read file with GT bounding boxes
        # header: ImgNo#.ppm;#leftCol#;##topRow#;#rightCol#;#bottomRow#;#ClassID#
        gt_images = pd.read_csv((self.dataset_path + 'gt.txt'), delimiter=';', header=None,
                                names=['filename'], usecols=[0])
        gt_images = gt_images.drop_duplicates()
        gt_images = gt_images.sort_values(by=['filename'])

        # read all images from the folder
        all_images = []
        images = glob.glob(self.dataset_path + '*.ppm')
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

        # print("All images\n", all_images.describe())
        # print("Positive images\n", gt_images.describe())
        # print("Negative images\n", negative_images.describe())
        print("We will slice {} negative images!".format(len(negative_images.index)))
        print("It can take a while ...")

        # create sliding window object
        sliding_window_parameters = {
            'x_win_len': 100,
            'y_win_len': 100,
            'x_increment': 60,
            'y_increment': 60,
            'svc_path': None,
            'scaler_path': None
        }
        sw = SlidingWindow(sliding_window_parameters)

        # Iterate through all images and for each one run sliding window to get slices. Then append it to slices list.
        slices = []
        for _, image in tqdm(negative_images.iterrows()):
            img = cv2.imread(self.dataset_path + image['filename'])
            sw.update_frame(img)
            for slice in sw.generate_slices():
                # imS = cv2.resize(slice, (500, int(slice.shape[0] * 500 / slice.shape[1])))
                # cv2.imshow(image['filename'], imS)
                # cv2.imshow("slice", slice)
                # cv2.waitKey(1)
                slices.append(slice)

        print("Extracted {} slices from negative images.".format(len(slices)))
        return slices
