import cv2
from termcolor import colored
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve


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
    This function plots images. It is used in printing HOG features visualisation
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


def print_executed():
    """
    This method prints time of execution of block
    :return: string with execution time
    """
    now = datetime.datetime.now()
    time = str(now.hour)+":"+str(now.minute)+":"+str(now.second)
    print("\n", colored("EXECUTED BLOCK AT "+time, 'green'))


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def get_hog_parameters():
    """
    This method returns global parameters for HOG descriptor
    :return: parameters of HOG descriptor
    """
    return {
        'color_model': 'hsv',  # hls, hsv, yuv, ycrcb
        'svc_input_size': 40,
        'number_of_orientations': 10,  # 6 - 12
        'pixels_per_cell': 8,
        'cells_per_block': 5,
        'do_transform_sqrt': True
    }