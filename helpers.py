import cv2
import matplotlib.pyplot as plt
from termcolor import colored
import datetime


def convert(frame, src_model="rgb", dest_model="hls"):
    """
    This function converts colorspace of image
    :param frame: frame of video or just image
    :param src_model: source color space model
    :param dest_model: destination color space model
    :return: converted image
    """
    if src_model == "rgb" and dest_model == "hsv":
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    elif src_model == "rgb" and dest_model == "hls":
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS)
    elif src_model == "rgb" and dest_model == "yuv":
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)
    elif src_model == "rgb" and dest_model == "ycrcb":
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YCR_CB)
    elif src_model == "hsv" and dest_model == "rgb":
        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2RGB)
    elif src_model == "hls" and dest_model == "rgb":
        frame = cv2.cvtColor(frame, cv2.COLOR_HLS2RGB)
    elif src_model == "yuv" and dest_model == "yuv":
        frame = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB)
    elif src_model == "ycrcb" and dest_model == "ycrcb":
        frame = cv2.cvtColor(frame, cv2.COLOR_YCR_CB2RGB)
    else:
        raise Exception('ERROR:', 'src_model or dest_model not implemented')

    return frame


def show_images(imgs, per_row=3, per_col=2, W=10, H=5, tdpi=80):
    """
    This function plots images
    :param imgs: array of images
    :param per_row: num of images per row
    :param per_col: num of images per column
    :param W: width of image
    :param H: height of image
    :param tdpi: dpi of image
    """
    fig, ax = plt.subplots(per_col, per_row, figsize=(W, H), dpi=tdpi)
    ax = ax.ravel()

    for i in range(len(imgs)):
        img = imgs[i]
        ax[i].imshow(img)

    for i in range(per_row * per_col):
        ax[i].axis('off')
    plt.show()

"""
DO POPRAWY CHYBA!!!!!!!!
"""
def box_boundaries(box):
    """
    This function gives back coordinates of bounding box
    :param box:
    :return: coordinates of bounding box
    """
    x1, y1 = box[0], box[1]
    x2, y2 = box[1] + box[2], box[1] + box[2]  # box[0]
    return x1, y1, x2, y2


def put_boxes(frame, boxes, color=(255, 0, 0), thickness=10):
    """
    This function writes out a bounding box on the image
    :param frame: frame of video or just image
    :param boxes: region of interest
    :param color: color of bounding box
    :param thickness: thickness of bounding box
    :return: image with bounding box
    """
    out_img = frame.copy()
    for box in boxes:
        x1, y1, x2, y2 = box_boundaries(box)
        cv2.rectangle(out_img, (x1, y1), (x2, y2), color, thickness)

    return out_img


def print_executed():
    """
    This method prints time of execution of block
    :return: string with execution time
    """
    now = datetime.datetime.now()
    time = str(now.hour)+":"+str(now.minute)+":"+str(now.second)
    print("\n", colored("EXECUTED BLOCK AT "+time, 'green'))
