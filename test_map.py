"""
This script is fully based on evaluation method proposed by ÁlvaroArcos-GarcíaJuan A.Álvarez-GarcíaLuis M.Soria-Morillo
as an attachment to the paper: "Evaluation of deep neural networks for traffic sign detection systems"
Below link to full repository of the project:
https://github.com/aarcosg/traffic-sign-detection#running-on-new-images
"""


import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np
import pandas as pd
import copy


def filter_detections_by_width(detections, min_w, max_w):
    """
    This method filters detections by given width and saves gt bboxes, which applies to given boundaries
    :param detections: list of detections
    :param min_w: minimum width
    :param max_w: maximum width
    :return: filtered list of detections
    """
    print(min_w, max_w)
    fdetects = []
    n_gt_bboxes = 0
    for d in detections:
        d_aux = copy.deepcopy(d)
        j = 0
        for i, bbox in enumerate(d['gt_bboxes']):
            w = bbox[2] - bbox[0]
            if w < min_w or w >= max_w:
                d_aux['gt_bboxes'].pop(i - j)
                d_aux['gt_classes'].pop(i - j)
                j += 1
        n_gt_bboxes += len(d_aux['gt_bboxes'])
        if len(d_aux['gt_bboxes']) > 0:  #####
            fdetects.append(d_aux)
    print(n_gt_bboxes)
    return fdetects


def intersection(bbgt, bb):
    """
    This method computes the intersection over ground truth bounding box and detected bounding box
    :param bbgt: ground truth bounding box
    :param bb: detected bounding box
    :return: list with intersections between bb and gtbb
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    ixmin = max(bbgt[0], bb[0])
    iymin = max(bbgt[1], bb[1])
    ixmax = min(bbgt[2], bb[2])
    iymax = min(bbgt[3], bb[3])

    # compute the area of intersection rectangle
    iw = max(ixmax - ixmin, 0)
    ih = max(iymax - iymin, 0)
    inters_area = iw * ih

    return inters_area


def union(bbgt, bb, intersArea):
    """
    This method computes the unions over ground truth bounding box and detected bounding box
    :param bbgt: ground truth bounding box
    :param bb: detected bounding box
    :param intersArea: intereection between bb ang gtbb
    :return: list with unions between bb all gtbb
    """
    # compute the area of both the prediction and ground-truth rectangles
    boxgtArea = (bbgt[2] - bbgt[0]) * (bbgt[3] - bbgt[1])
    boxArea = (bb[2] - bb[0]) * (bb[3] - bb[1])
    unionArea = boxgtArea + boxArea - intersArea
    return unionArea


def compute_iou(bbgt, bb):
    """
    Returns the intersection over union of two rectangles, a and b, where each is an array [x,y,w,h]
    :param bbgt: ground truth bounding box
    :param bb: detected bounding box
    :return: intersection over union
    """
    overlaps = np.zeros(len(bbgt))
    for i, gtBbox in enumerate(bbgt):
        inters = float(intersection(gtBbox, bb))
        uni = union(gtBbox, bb, inters)
        iou = inters / uni
        overlaps[i] = iou
    ioumax = np.max(overlaps)
    jmax = np.argmax(overlaps)
    return ioumax, jmax


def voc_ap(rec, prec, use_07_metric=False):
    """
    Compute VOC AP given precision and recall. If use_07_metric is true, uses the VOC 07 11
     point method (default:False). Usage: ap = voc_ap(rec, prec, [use_07_metric])
    :param rec: recalls list
    :param prec: precisions list
    :param use_07_metric:
    :return: average precision
    """
    if use_07_metric:
        # 11 point metric
        # http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham10.pdf (page 313)

        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation (from VOC 2010 challenge)
        # http://host.robots.ox.ac.uk/pascal/VOC/voc2012/devkit_doc.pdf (page 12)

        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def match_gts_and_compute_prec_recall_ap(cls, detections, category_index, iouthresh):
    """
    A bounding box reported by an algorithm is considered correct if its area intersection over union with a ground
    truth bounding box is beyond X%. If a lot of closely overlapping bounding boxes hitting on a same ground truth,
    only one of them is counted as correct, and all the others are treated as false alarms.
    :param cls:
    :param detections:
    :param category_index: python dictionary with indices of classes in categories dict. e.g.: {1: categories[0]}
    :param iouthresh: threshold of intersection over union, above which detection is treated as truth
    :return:
        -rec: recall
        -prec: precision
        -ap: average precision
    """

    print('IoU threshold set to: {:.2f}'.format(iouthresh))
    GT_OBJECTS = {}
    BB = []
    BB_im_ids = []
    BB_confidences = []
    n_gt_bboxes = 0

    for dId, d in enumerate(detections):
        BBGT = []
        for i in range(len(d['gt_bboxes'])):
            if d['gt_classes'][i] == cls:
                BBGT.append(d['gt_bboxes'][i])
                n_gt_bboxes += 1
        GT_OBJECTS[d['file']] = {
            'bboxes': np.asarray(BBGT),
            'detected?': [False] * len(BBGT)
        }
        for i in range(len(d['pred_bboxes'])):
            if d['pred_classes'][i] == cls:
                BB.append(d['pred_bboxes'][i])
                BB_im_ids.append(d['file'])
                BB_confidences.append(d['confidences'][i])

    if n_gt_bboxes == 0:
        return None, None, None

    BB = np.asarray(BB)
    BB_confidences = np.asarray(BB_confidences)

    # sort by confidence
    if len(BB) > 0:
        sorted_ind = np.argsort(-BB_confidences)
        sorted_scores = np.sort(-BB_confidences)
        BB = BB[sorted_ind, :]
        BB_im_ids = [BB_im_ids[x] for x in sorted_ind]

    num_detections = len(BB_im_ids)
    tp = np.zeros(num_detections)
    fp = np.zeros(num_detections)
    avg_overlap = []

    for d in range(num_detections):
        gt_info = GT_OBJECTS[BB_im_ids[d]]
        bb = BB[d, :].astype(float)
        BBGT = gt_info['bboxes'].astype(float)
        ioumax = -np.inf

        if BBGT.size > 0:
            # compute intersection over union
            ioumax, jmax = compute_iou(BBGT, bb)
        if ioumax > iouthresh:
            if not gt_info['detected?'][jmax]:
                tp[d] = 1.  # true positive
                gt_info['detected?'][jmax] = 1
                avg_overlap.append(ioumax)
            else:
                fp[d] = 1.  # false positive (multiple detection)
        else:
            fp[d] = 1.  # false positive

    avg_overlap = np.array(avg_overlap) if len(avg_overlap) > 0 else np.array([0])

    # compute precision recall
    fp = np.cumsum(fp) if len(fp) > 0 else np.array([0])
    tp = np.cumsum(tp) if len(tp) > 0 else np.array([0])

    fn = n_gt_bboxes - tp[-1]

    rec = tp / np.maximum(tp + fn, np.finfo(np.float64).eps)
    # avoid divide by zero in case the first detection matches a difficult ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    ap = voc_ap(rec, prec)

    print("TP: {}".format(tp[-1]))
    print("FP: {}".format(fp[-1]))
    print("FN: {}".format(fn))
    class_name = category_index[cls]['name']
    print('Avg. overlap for {} = {:.4f}'.format(class_name, np.mean(avg_overlap)))

    print('Precision for {} = {:.4f}'.format(class_name, prec[-1]))
    print('Recall for {} = {:.4f}'.format(class_name, rec[-1]))

    return rec, prec, ap


def plot_precision_recall(prec, recall, ap):
    """
    This method plots precision recall curve for each class
    :param prec: precision list
    :param recall: recall list
    :param ap: average precision value
    """
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [recall[-1]]))
    mpre = np.concatenate(([prec[0]], prec, [0.]))
    # Plot Precision-Recall curve
    plt.clf()
    plt.plot(mrec, mpre, lw=2, color='navy',
             label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall: AP={0:0.2f}'.format(ap))
    plt.legend(loc="lower left")
    plt.show()


def plot_full_precision_recall(data, path, name="figure"):
    """
    This method plots full precision recall curve for all data
    :param data: list with all data -  class names, precisions for each, recalls for each, average precision for each
    :param path: path to save figure
    :param name: name of figure
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    lines = ["-", "--", "-.", ":"]
    linecycler = cycle(lines)
    for d in data:
        recall = d['recall']
        prec = d['precision']
        ap = d['ap']
        cls = d['class']
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], recall, [recall[-1]]))
        mpre = np.concatenate(([prec[0]], prec, [0.]))
        ax.plot(mrec, mpre, next(linecycler), label='{} (AP = {:.2f}%)'.format(cls, ap * 100))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc="lower left")
    plt.draw()
    plt.tight_layout()
    plt.savefig(path + name + '.pdf', format='pdf')
    plt.show()
    plt.clf()


def compute_mean_average_precision(detections, categories, category_index, path, name="figure"):
    """
    For each class, we compute average precision (AP). This score corresponds to the area under the precision-recall
    curve. The mean of these numbers is the mAP.
    :param detections: python list of objects with fields: class_given_obj, confidences, bboxes
    :param categories: python dictionary with number of classes and its labels, e.g.: [{'id': 1, 'name': 'sign'}]
    :param category_index: python dictionary with indices of classes in categories dict. e.g.: {1: categories[0]}
    :param path: python string, path to save figure in
    :param name: python string, name of figure to save as pdf
    :return: mean average precision (float)
    """

    results = []
    plot_data = []
    aps = []

    for category in categories:
        class_name = category['name']
        rec, prec, ap = match_gts_and_compute_prec_recall_ap(category['id'], detections, category_index,
                                                             iouthresh=0.1)
        if rec is None:
            continue
        results.append({'class': class_name, 'precision': prec[-1], 'recall': rec[-1], 'ap': ap})
        plot_data.append({'class': class_name, 'precision': prec, 'recall': rec, 'ap': ap})
        if ap is not None:
            aps += [ap]
            print('AP for {} = {:.4f}'.format(class_name, ap))
        plot_precision_recall(prec, rec, ap)
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    mAP = np.mean(aps)
    df = pd.DataFrame.from_records(results, columns=('class', 'precision', 'recall', 'ap'))
    print(df)
    plot_full_precision_recall(plot_data, path, name)
    return mAP
