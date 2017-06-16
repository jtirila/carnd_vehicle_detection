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

It should be noted that I was a little bit too ambitious about the project structure, and the attempt to 
basically restructure everything ended up taking several days of debugging and refactoring, only to finally 
basically come back to something rather close to the example code. Anyhow, it was a nice learning experience 
both in computer vision/machine learning and managing complexity in a software project. 


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

* Some Python 3.x version. The project has been written on Python 3.5. I am not aware if compatibility varies between 
  different minor versions of Python 3. Possibly package dependencies could cause difficulties with some combinations.  
* All the Python packages needed to run the code, including Numpy, OpenCV and MatPlotLib. 
  The easisest way to get both Python and the packages up and running is probably to install the 
  [Udacity started kit](https://github.com/udacity/CarND-Term1-Starter-Kit). 
* A suitable set of training images to train the vehicle detection classifier. These files will not be included in the 
  repository. The images have to be placed in the `images` subdirectory of the top level of the 
  project, with vehicle images place in subdirectories of `vehicle` and non-vehicle images in subdirectories of 
  `nonvehicle` within the image directory. 
  
  Or, alternatively, if you wish to take care of 
  loading the images yourself, the `classify/svm_classifier.py` file documents how to provide 
  the training data directly as Python data structures. 
* The project video is also omitted from this repository. You should find 
  [the one used for the project](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/project_video.mp4) 
  or an equivalent video. Probably there are some requirements on the type of video but I don't know them as of now. 
  
  By default, there needs to be a video called `project_video.mp4` in the top level directory. 
  See `detect_vehicles.py` for how you can provide a different filename as a parameter. 
* You probably also need to have `ffmpeg` installed. Not sure about this. If the processing fails, you 
  can try installing `ffmpeg`. 

#### Launching the vehicle detector

With the prerequisities covered, you can simply run the vehicle detector script by 

```
sh detect_vehicles.py
```

in the directory where the file is located, or by launching the Python interpreter and issuing the following commands. 

```python
from carnd_vehicle_detection.detect_vehicles import detect_vehicles
detect_vehicles
```

This will by default result in a rather long processing pipeline. The package contains utils for creating 
subclips of the video in the `utils/one_off_scripts` directory and with there are commented-out references to these 
shorter passages of the video in `detect_vehicles.py`. What exactly to do is currently uncodumented but not too 
difficult to figure out.  
To run the code

#### The result

Once the script is done, by default is has produced a video called `transformed.mp4` in the top level directory.

##### Training images

You will need a set of both vehicle and non-vehicle images to train the classifier. These images are 
not included in the repository. The easiest way is to place the images as follows, relative to the 
top level of the repository. 

## The project 

### Goals



### My approach



### Overview of my solution in terms of the rubric points 



#### First point

#### Data preprocessing

#### Feature extraction

##### FIXME: some color histogram stuff maybe
##### Extracting the HOG features, the first attempt

The histogram of oriented gradients is a nice approach to indclude some generic shape templates in the feature vector.  
The method determines (discrete) gradient directions an magnitudes for each pixel in the image, and then bins them 
according to the chosen cell size and blocking scheme. 


The result is a histogram of gradient directions for each cell, 
weighted by the magnitudes, so that the dominant gradient directions are extracted. 
This is illustrated in the image below. 

FIXME: include image of hog features. 

The number of direction bins to use can be provided as a parameter. I had played around with different numbers
of bins in the labs preceding the project, but did not have any systematic hunch of what works and why. 

As my pipeline started to grow rather heavy computation-wise, I figured I need to settle on a rather small
amount of bins so the processing would not take too long. I did not observe any systematic difference between different 
 bin numbers, so I ended up using 9 as suggested in the labs. 
 
As for the number of cells per block, I used the suggested 8 x 8. The rationale for this can be summarized as follows:  

* I figured it would be nice to have training image sizes, and later candidate image sizes, to be multiples of 
  the cell width. This way, especially in the valuable training data, no information needs to be discarded in the 
  HOG feature vector due to a too narrow band of pixels at the right or at the bottom of an image. 
* Another scale related thing is that for each 64x64 sized image, I wanted to have a reasonable number of cells.  
  Going below 8 seemed like some of the bigger-scale spatial features would be lost. 
  On the other hand, to be able to have more than 8x8 cells, the pixels per cell value would have to be set so 
  low I suspected the cell values would become unstable. To be able to form a meanigful histogram of 
  gradient orientations, it seems to me one needs at least a few dozen samples, and this from this point of view
  also 8x8 pixels seems like a reasonable choice. 

The cells per block parameter was left at 2. My reasoning was that again, the training images being 64x64, 
one would not want to normalize over a too large area to preserve enough of the local features. Not sure if this 
reasoning makes sense, but also [this presentation on pedestrian detection](https://www.youtube.com/watch?v=7S5qXET179I)
suggests 8 pixels per cell and 2 cells per block should be a nice compromise. There is a slide concerning the 
miss rates using various combinations of the parameters, and a choice of 8 and 2 yields the smallest miss rate. 

###### The Relationship Between Moving Windows and HOG Features

When planning my algorithm, I first tried to implement a method of my own for the window search, kind of thinking 
of the scaling in reverse order compared to the code snippet provided as an example. I tried to keep the 
frames in their original size, and choose the candidate window locations and sizes so as to be able to extract the same 
HOG information as that from the 64x64 training images. The method to do this is rather convoluted, 
and contains a lot of intricate details on how exactly the cells should be placed at least if the 
features were to mach the chosen window location exactly. I will not describe the approach here in any 
more detail as I am unaware if it even has any benefits to the standard one. It was a nice learning experience 
anyway. The code is still included in the repository and maybe I try to get it to work again. 

As for the standard approach, the `find_cars` function provided in the project instructions essentially  
scales the whole frame according to the provided scale parameter, and then looks for car detections in 
candidate windows of size 64x64 so the processing is quite straightforward. The only bit about this 
that is rather convoluted is the computation of the various mapped x and y locations, step sizes etc. 

##### Spatial Features an Color Histograms

During the project, I experimented both with including all the three feature sets both with including all the three 
feature sets both with including all the three feature sets both with including all the three feature sets in the 
final feature vector, and leaving the color histogram and spatial features out. Here are some thoughts on each of 
these feature categories and why they may not be as important for vehicle detection as the HOG features. 

###### Spatial features

The spatial features vector is just a compressed version of the original image. While the main characteristics of e.g.  
a vehicle are still preserved in a, say, 32x32 pixel image, I doubt the usefulness of such a feature vector
depends heavily on the type of classifier used. As I ended up using an SVM classifier for the project,  
I am not really sure if any linear decision surface would be able to use the compressed version of an 
image very efficiently.

If I were to use a convolutional neural network, for example, the situation might be different, though. 

###### Color histogram

As for the color histogram, I think its usefulness in a SVM classifier context may be a bit better justified 
than that of the spatial feature vector. This is especially true for alternative color spaces that 
are probably better able to distinguish between saturated "artificial" colors used in vehicles, and 
the more washed out background of a road environment. Care should be taken though so the classifier 
does not go totally wild in a colorful city environment. 

Even for the color histograms, my findings did not suggest an immediate usefulness, but in a real-world 
detection problem, I would probably still consider incluging them due to the potential effects mentioned above. 

However, in conclusion regarding both the spatial and color histogram features, I had a hard time obtaining a stable 
vehicle detection so I ended up including all of the suggested features
anyway, hoping they would provide even a tiny bit of assistance in detecting vehicles in more challenging 
portions of the video.  




#### Training the classifier


#### ETC
#### The moving window search, tracking vehicles and removing false positives

TODO: These are covered in one section because the technique to implement these two things 
is higly overlapping.



### The solution

TODO: maybe  upload the video to YouTube and include a (image) link to the video in the writeup, as 
per https://stackoverflow.com/a/16079387



### Discussion
