#!/usr/bin/env python
"""
This a modified version of the object detector available from caffe.
A good starting point thus is to take a look at the original version:
http://nbviewer.ipython.org/github/BVLC/caffe/blob/master/examples/detection.ipynb
The version has been modified by Alessandro Ferrari (alessandroferrari87@gmail.com).

detect.py is an out-of-the-box windowed detector callable from the command line.

By default it configures and runs the Caffe reference ImageNet model.
Note that this model was trained for image classification and not detection,
and finetuning for detection can be expected to improve results.

Bing code is available at https://github.com/alessandroferrari/BING-Objectness .

Example usage:

python detect.py --crop_mode=bing 
--pretrained_model=/path/to/caffe/models/bvlc_reference_rcnn_ilsvrc13/bvlc_reference_rcnn_ilsvrc13.caffemodel 
--model_def=/path/to/caffe/models/bvlc_reference_rcnn_ilsvrc13/deploy.prototxt 
--mean_file=/path/to/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy --gpu 
--raw_scale=255 --weights_1st_stage_bing /path/to/BING-Objectness/doc/weights.txt 
--sizes_idx_bing /path/to/BING-Objectness/doc/sizes.txt 
--weights_2nd_stage_bing /path/to/BING-Objectness/doc/2nd_stage_weights.json 
--num_bbs_final 2000 --detection_threshold 0.1 /path/to/pictures/image.jpg 
/path/to/results/output.jpg /path/to/caffe/data/ilsvrc12/det_synset_words.txt
"""
import os
import cv2
import time
import json
import caffe
import argparse
import numpy as np
from random import randint
from skimage.io import imread
from detector import Detector, resize_image

CROP_MODES = ['bing']
COORD_COLS = ['ymin', 'xmin', 'ymax', 'xmax']

def nms_detections(dets, overlap=0.3):
    """
    Non-maximum suppression: Greedily select high-scoring detections and
    skip detections that are significantly covered by a previously
    selected detection.

    This version is translated from Matlab code by Tomasz Malisiewicz,
    who sped up Pedro Felzenszwalb's code.

    Parameters
    ----------
    dets: ndarray
        each row is ['xmin', 'ymin', 'xmax', 'ymax', 'score']
    overlap: float
        minimum overlap ratio (0.3 default)

    Output
    ------
    dets: ndarray
        remaining after suppression.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    ind = np.argsort(dets[:, 4])

    w = x2 - x1
    h = y2 - y1
    area = (w * h).astype(float)

    pick = []
    while len(ind) > 0:
        i = ind[-1]
        pick.append(i)
        ind = ind[:-1]

        xx1 = np.maximum(x1[i], x1[ind])
        yy1 = np.maximum(y1[i], y1[ind])
        xx2 = np.minimum(x2[i], x2[ind])
        yy2 = np.minimum(y2[i], y2[ind])

        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)

        wh = w * h
        o = wh / (area[i] + area[ind] - wh)

        ind = ind[np.nonzero(o <= overlap)[0]]

    return dets[pick, :]

def main(argv):
    pycaffe_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    # Required arguments: input and output.
    parser.add_argument(
        "image_file",
        help="Input image filename."
    )
    parser.add_argument(
        "output_image_file",
        help="Output image filename. Format depends on extension."
    )
    # Optional arguments.
    parser.add_argument(
        "--model_def",
        default=os.path.join(pycaffe_dir,
                "../models/bvlc_reference_caffenet/deploy.prototxt.prototxt"),
        help="Model definition file."
    )
    parser.add_argument(
        "--pretrained_model",
        default=os.path.join(pycaffe_dir,
                "../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"),
        help="Trained model weights file."
    )
    parser.add_argument(
        "--crop_mode",
        default="bing",
        choices=CROP_MODES,
        help="How to generate windows for detection."
    )
    parser.add_argument(
        "--weights_1st_stage_bing_fn",
        default=None,
        help="Weights generated for the first stage of bing."
    )
    parser.add_argument(
        "--sizes_idx_bing_fn",
        default=None,
        help="Indeces of the active sizes for bing."
    )
    parser.add_argument(
        "--weights_2nd_stage_bing_fn",
        default=None,
        help="Weights generated for the second stage of bing."
    )
    parser.add_argument(
        "--num_bbs_psz_bing",
        default=130,
        type=int,
        help="Number of bounding boxes per size index in bing."
    )
    parser.add_argument(
        "--num_bbs_final_bing",
        type=int,
        default=1500,
        help="Final number of bounding boxes candidates in bing."
    )
    parser.add_argument(
        "--gpu",
        action='store_true',
        help="Switch for gpu computation."
    )
    parser.add_argument(
        "--mean_file",
        default=os.path.join(pycaffe_dir,
                             'caffe/imagenet/ilsvrc_2012_mean.npy'),
        help="Data set image mean of H x W x K dimensions (numpy array). " +
             "Set to '' for no mean subtraction."
    )
    parser.add_argument(
        "--input_scale",
        type=float,
        help="Multiply input features by this scale to finish preprocessing."
    )
    parser.add_argument(
        "--raw_scale",
        type=float,
        default=255.0,
        help="Multiply raw input by this scale before preprocessing."
    )
    parser.add_argument(
        "--channel_swap",
        default='2,1,0',
        help="Order to permute input channels. The default converts " +
             "RGB -> BGR since BGR is the Caffe default by way of OpenCV."
    )
    parser.add_argument(
        "--context_pad",
        type=int,
        default='16',
        help="Amount of surrounding context to collect in input window."
    )
    parser.add_argument(
        "--detection_threshold",
        type=float,
        default=0.1,
        help="Score threshold for determining positive detection from the convolutional network."
    )
    parser.add_argument(
        "--reference_edge",
        type=float,
        default=512.0,
        help="Size to which rescale the maximum edge of the image. The rescaling mantain the proportions in the image."
    )
    parser.add_argument(
        "synset_file",
        help="Filename that contains the synset that corresponds to the classes of the network."
    )
    
    args = parser.parse_args()

    synset_dict = dict()
    f = open(args.synset_file)
    counter = 0
    while True:
        s = f.readline()
        if s=='':
            break
        name = s[s.find(" ")+1:].replace("\n","")
        synset_dict[counter] = name
        counter = counter + 1
    f.close()

    if not args.weights_1st_stage_bing_fn is None:
        if not os.path.exists(args.weights_1st_stage_bing_fn):
            print "Error the path specified for weights_1st_stage_bing_fn %s does not exist."%args.weights_1st_stage_bing_fn
            sys.exit(2)
        weights_1st_stage_bing = np.genfromtxt(args.weights_1st_stage_bing_fn, delimiter=',', dtype=np.float32)
    else:
        weights_1st_stage_bing = None
    
    if not args.weights_2nd_stage_bing_fn is None:
        if not os.path.exists(args.weights_2nd_stage_bing_fn):
            print "Error: the path specified for weights_2nd_stage_bing_fn %s does not exist."%args.weights_2nd_stage_bing_fn
            sys.exit(2)
        f = open(args.weights_2nd_stage_bing_fn,"r")
        weights_2nd_stage_bing_str = f.read()
        f.close()
        weights_2nd_stage_bing = json.loads(weights_2nd_stage_bing_str)
    else:
        weights_2nd_stage_bing = None

    if not args.sizes_idx_bing_fn is None:
        if not os.path.exists(args.sizes_idx_bing_fn):
            print "Error: the path specified for sizes_idx_bing_fn %s does not exists."%args.sizes_idx_bing_fn
            sys.exit(2)
        sizes_idx_bing = np.genfromtxt(args.sizes_idx_bing_fn, delimiter=',').astype(np.int32)
    else:
        sizes_idx_bing = None
        
    mean, channel_swap = None, None
    if args.mean_file:
        mean = np.load(args.mean_file)
    if args.channel_swap:
        channel_swap = [int(s) for s in args.channel_swap.split(',')]

    # Make detector.
    detector = Detector(args.model_def, args.pretrained_model,
            gpu=args.gpu, mean=mean,
            input_scale=args.input_scale, raw_scale=args.raw_scale,
            channel_swap=channel_swap,
            context_pad=args.context_pad,
            weights_1st_stage_bing = weights_1st_stage_bing, 
            sizes_idx_bing = sizes_idx_bing,
            weights_2nd_stage_bing = weights_2nd_stage_bing, 
            num_bbs_psz_bing = args.num_bbs_psz_bing, 
            num_bbs_final_bing = args.num_bbs_final_bing)

    if args.gpu:
        print 'GPU mode'

    t = time.time()

    image_fn = args.image_file
    image = imread(image_fn)
    image = resize_image(image, args.reference_edge)

    if args.crop_mode == "bing":
        detections, predictions = detector.detect_bing(image)
    else:
        print "Crop mode is not supported!"
        sys.exit(1)
        
    print("Processed {} windows in {:.3f} s.".format(len(detections),
                                                     time.time() - t))
    
    predictions[predictions<args.detection_threshold] = 0
    
    indices = np.nonzero(predictions)
    
    classes = np.unique(indices[1])

    resulting_bbs = []
    for cls in classes:
        mask = indices[1]==cls
        bbs_idx = indices[0][mask]
        bbs_for_nms = [ np.array( (detections[idx]["window"][1], detections[idx]["window"][0], detections[idx]["window"][3], detections[idx]["window"][2], predictions[idx,cls]) )  for idx in bbs_idx]
        for detection in list(nms_detections(np.array(bbs_for_nms))):
            score = detection[4]
            bb = detection[:4].astype(int)
            resulting_bbs.append((bb,score,synset_dict[cls]))
        
    image = cv2.cvtColor(image, cv2.cv.CV_RGB2BGR)
    
    for bb, det_score, class_name in resulting_bbs:
        clr = (randint(0,255), randint(0,255), randint(0,255))
        cv2.putText(image,"%s"%class_name, (int(bb[0])+ 10,int(bb[1]) + 10), cv2.FONT_HERSHEY_TRIPLEX, 0.6, clr)
        cv2.rectangle(image,(int(bb[0]),int(bb[1])),(int(bb[2]),int(bb[3])),color=clr)

    cv2.imwrite(args.output_image_file, image)

    cv2.imshow("Detection result",image)
    cv2.waitKey()

if __name__ == "__main__":
    import sys
    main(sys.argv)
