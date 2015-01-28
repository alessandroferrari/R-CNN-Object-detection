# R-CNN-Object-detection
Python-caffe simplified implementation of the R-CNN object detection method. 

I have taken as a starting point the caffe ipython-notebook available at http://nbviewer.ipython.org/github/BVLC/caffe/blob/master/examples/detection.ipynb . 
Thus, for getting the model and all the other files I recommend to take a look there.

The tutorial requires the use of matlab code for calculating selective search bounding boxes. This make it hard to use and slow. 

Thus, I have implemented bounding boxes proposal with a pythonized BING that I have implemented in another repository.  

I have also changed the interaction with the script so that the result is a nicer demo. At the end of the execution, you get an immediate visual response of the results of the classifier on the selected image.

Objects detection takes approximately 15 seconds for image on an Intel i7 4930K processor and an Nvidia Titan Black GPU.

Dependencies:

-Python
-Caffe
-numpy
-opencv
-scikit-image
-BING-Objectness (available at https://github.com/alessandroferrari/BING-Objectness )

For explaination on the usage:

- Move to the repository folder on your command line;
- type 'cd source'
- type 'python detect.py -h' for getting complete synopsys about the program.

Example of usage:
python detect.py --crop_mode=bing 
--pretrained_model=/path/to/caffe/models/bvlc_reference_rcnn_ilsvrc13/bvlc_reference_rcnn_ilsvrc13.caffemodel 
--model_def=/path/to/caffe/models/bvlc_reference_rcnn_ilsvrc13/deploy.prototxt 
--mean_file=/path/to/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy --gpu 
--raw_scale=255 --weights_1st_stage_bing /path/to/BING-Objectness/doc/weights.txt 
--sizes_idx_bing /path/to/BING-Objectness/doc/sizes.txt 
--weights_2nd_stage_bing /path/to/BING-Objectness/doc/2nd_stage_weights.json 
--num_bbs_final 2000 --detection_threshold 0.1 /path/to/pictures/image.jpg 
/path/to/results/output.jpg /path/to/caffe/data/ilsvrc12/det_synset_words.txt

Acknowledgments:
Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik and to the Caffe folks.

Authors:
Alessandro Ferrari - alessandroferrari87@gmail.com

Licensing:
gpl 3.0

Enjoy.