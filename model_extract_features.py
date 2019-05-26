from tsr_hog_extractor import HOGExtractor
import numpy as np
import cv2
import glob
from tqdm import tqdm
import pandas as pd
from tsr_helpers import get_hog_parameters


def extract_positive_features(type_of_signs, extractor):
    """
    This method extract features from given type of signs (type accurate to Vienna convention 1968).
    :param type_of_signs: type of sign - single char, e.g. 'b'
    :param extractor: HOH extractor
    :return: In the end function saves np array with features to file
    """

    # read file which describe positive dataset
    positive_data_descr = pd.read_csv('dataset/positive/folders_classes.csv', delimiter=';', dtype=object)
    positive_data_descr = positive_data_descr.drop('opis', axis=1)

    # select only signs of certain class
    positive_data_descr = positive_data_descr.loc[(positive_data_descr['typ'] == type_of_signs)]
    print("Do extraction from folders below:\n{}".format(positive_data_descr))

    # initialise list for images
    positive_images = []

    # for each selected folder
    for folder in positive_data_descr['folder']:
        # read all images from it and sort it
        paths = sorted(glob.glob('./dataset/positive/' + folder + '/*.ppm'))
        # read csv which contains bounding boxes of traffic sign
        csv = pd.read_csv('./dataset/positive/' + folder + '/GT-' + folder + '.csv', delimiter=';')
        print("\nReading images from folder: ", folder)
        for path in tqdm(paths):
            # read name of the image
            name = path.split('/')[4]
            # search for selected image in the csv and flatten that row to list
            row = (csv.loc[csv['Filename'] == name]).values.tolist()
            # read image and cut ROI with traffic sign, then add to list
            image = cv2.imread(path)
            if image.shape[0] >= 25 or image.shape[1] >= 25:
                image = image[row[0][3]:row[0][5], row[0][4]:row[0][6]]
                positive_images.append(image)

    positive_images = np.asarray(positive_images)

    print("\nExtracting features from traffic signs...")

    # initialise list for calculated features
    positive_features = []

    # for each loaded image
    for img in tqdm(positive_images):
        positive_features.append(extractor.features(img))

    positive_features = np.asarray(positive_features)
    np.save('dataset/extracted_features/positive_features_' + type_of_signs, positive_features)

    print("\nAll done for class: {}!\n".format(type_of_signs))


def extract_negative_features(folder, extractor):
    """
    This method extract features from nonsigns.
    :param folder: path to folder with negative images
    :param extractor: HOH extractor
    :return: In the end function saves np array with features to file
    """

    # initialise list for images
    negative_images = []

    # read all images from folder
    paths = (glob.glob('./dataset/negative/' + folder + '/*.png'))
    print("\nReading images from folder: ", folder)
    for path in tqdm(paths):
        # read image
        image = cv2.imread(path)
        if image.shape[0] >= 25 or image.shape[1] >= 25:
            negative_images.append(image)

    negative_images = np.asarray(negative_images)

    print("\nExtracting features from non traffic signs in {}...".format(folder))

    # initialise list for calculated features
    negative_features = []

    # for each loaded image
    for img in tqdm(negative_images):
        negative_features.append(extractor.features(img))

    negative_features = np.asarray(negative_features)
    np.save('dataset/extracted_features/negative_features_' + folder, negative_features)

    print("\nAll done for class: {}!\n".format(folder))


# Initialise HOG extractor. Note, that its parameters should be equal for features of all sign types which are being
# used in SVM training.
extractor = HOGExtractor(get_hog_parameters())

# read file which describe types of signs and select its language
sign_description = pd.read_csv('dataset/sign_types.csv', delimiter=';')
sign_description = sign_description.drop('opis', axis=1)
print(sign_description.head(5), '\n')


types_of_signs = ['a', 'b', 'c', 'd']
for group in types_of_signs:
    extract_positive_features(group, extractor)

'''
folders = ['road', 'KITTI_extracted']
for folder in folders:
    extract_negative_features(folder, extractor)
'''
