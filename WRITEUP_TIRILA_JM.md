# Vehicle detection project, P5

FIXME: Heavily under construction. Everything is subject to change.  


## Notes on the submission

### Repository structure

For this project, I decided to use a conventional Python project hierarchy, with packages and modules arranged 
into a tree of directories and source code files grouped together according to related functionality. The 
structure of the project is a follows (omitting parts not essential fot this illustration): 

#### Hierarchy of packages and modules
```
|── README.md
├── WRITEUP_TIRILA_JM.md
├── carnd_vehicle_detection
│   ├── __init__.py
│   ├── classify
│   │   └── svm_classifier.py
│   ├── detect_vehicles.py
│   ├── image_transformations
│   │   ├── __init__.py
│   │   └── hog_transform.py
│   ├── image_traversal
│   │   ├── __init__.py
│   │   └── sliding_windows.py
│   └── preprocessing
│       ├── __init__.py
│       ├── extract_features.py
│       └── read_classification_training_data.py
├── images
├── project_video.mp4
└── unit_tests
```

The top-level `carnd_vehicle_detection` directory acts as a container for everything, including documentation 
(readme, writeup) and the included assets (video). A subdirectory of the same name contains just 
the Python code. 

**When mentioning code file paths in this writeup, the root of reference is always the innermost 
`carnd_vehicle_detection` directory.**





#### Unit tests

Even though writing a comprehensive set of unit tests would be overkill for a one-off project like this 
(and the included one is far from comprehensive), I tend to think of some kind of a test set 
also as a development tool, making it easier to repeatedly run smaller pieces of code, possibly using
smaller sets of data to process etc. 

Also, thingking about testability helps me reason about the internal API of a project. 
 
Due to these reasons, I have included a small set of unit tests in this submission. 

**As far as I am concerned, the test suite can be omitted from the review as all the real 
functionality resides outside of the test directory.**


### Running the code

#### Prerequisities

#### Running as a script

#### Importing, then running

## The project 

### Goals

### My approach

TODO: maybe something general about how I implemented the pipeline, if this is not clear just looking at the 
rubric points. 

### Overview of my solution in terms of the rubric points 


#### First point

#### Data preprocessing

#### Feature extraction
##### FIXME: some color histogram stuff maybe
##### Extracting the HOG features

###### The Relationship Between Moving Windows and HOG Features

For the training data, the HOG feature vector is quite straightforward: the vector needs to be extracted for each image 
anyway. When scanning an image for vehicle matches, the situation is different. The proposed solution will iterate over 
each frame using a multitude subimages (corresponding to "moving windows") as FIXME(described above|check that I have 
explained this somewhere). To scan the frame thoroughly enough, even at a single scale there will be a lot of overlap 
between the windows to avoid missing vehicles due to the effect of the windowing step. FIXME(See an exaggerated 
illustration of this effect in the figure below.|Make a figure with a very probable false negative because of too 
long steps). In a naive implementation, the HOG features would be extracted multiple times for each of these 
overlapping regions. 

This overhead in processing is further emphasized with the multi-scale search method outlined above. With e.g. three 
different scales of search windows. 

Due to the performance issue outlined above, I decided to extract the HOG features just once for every scale. However, 
this needs to be planned carefully together with the scaling and windowing algorithms due to the following effect: 

_The HOG feature vector needs to be of the same size independent of scale, and the semantics of each positional element
in the vector needs to stay consistent across scales._ In practice this means that the cell counts in each dimension 
 must be altered for the HOG transform so as to obtain a consistent amount of cells where orientation histograms 
are computed. This is somewhat hinted at in the example code provided for the project, but just for the sake of it, 
I decided to implement my own version of the scaling algorithm. 

The outline of my scaling algorithm is as follows. One should remember here that for the training data, the 
image size is 64x64 pixels and the pix_per_cell parameter value is 8 so the images are divided into 8x8 cells.  

* As 64x64 pixels corresponds to a rather small portion of the video frames, and hence vehicle detections pretty far 
  away from the camera, I figured there is no point going below 64x64 pixels for the windows. So, I'm using windows 
  sizes from 64x64 pixels upwards.  
* Upon choosing window sizes, to avoid any unwanted scaling effects, I wanted the new windows sizes to be exact 
  multiples of 8. This guarantees that all the windowed test images can be evenly divided into cells by the algotirhm 
  described below and there are now leftover cells or windows smaller than the others due to roundings because of 
  fractional division results.  
* Another natural requirement is that the windows are square in shape just like the training data.  
* Now, any window sizes from the sequence 64x64, 72x72 (9 by 9 cells), 80x80 (10 by 10 cells), 88x88 (11 by 11 cells) 
  etc. would satisfy the requirements above. I decided to first attempt with not too many scales, and the largest scale 
  being suitable for detecting vehicles just in front of the camera. With these requirements in mind, and after 
  evaluating the performance of the solution, I ended up using just three different scales: 64x64, 128x128 and 256x256 
  pixels. 
* Yet another requirement, to be able to pick pre-computed HOG feature subsets exactly at the correct positions, the 
  window movement step sizes must be multitudes of the scale-specific pix_per_cell values.  
  
With the scheme outlined above, for each scale I'll use a pix_per_cell values of 
new_win_widht / original_window_width * 8, guaranteed to be an even number and also guaranteed to produce a uniform 
cell grid on the windows with no leftover pixels or any other weirdness. 

For a specific scale, the HOG feature extraction using this scaled pix_per_cell count is now applied over all of the 
region of interest just once, and then for each individual window, 


There is another consideration with the formation of the HOG feature vectors: while the method of computing the HOG 
features for the training images (producing feature arrays of shape FIXME, collected directly as flattened feature 
vectors of size FIXME),  



**The inefficiency of extracting HOG features for each **

I initially experimented with extracting the 

Due to various 



#### The moving window search

#### Tracking vehicles and removing false positives

TODO: These are covered in one section because the technique to implement these two things 
is higly overlapping.
#### Training the classifier
#### ETC


### The solution

TODO: maybe  upload the video to YouTube and include a (image) link to the video in the writeup, as 
per https://stackoverflow.com/a/16079387



### Discussion
