# Vehicle detection project, P5

**Writeup by J-M Tirilä**

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

The repository contains code files not used in the actual submitted processing. This is indicated in the 
comments. The most important files the reviewers should pay attention to are `detect_vehicles.py` and 
`traverse_image/search_windows.py`. The rest mainly implement thin wrappers around OpenCV image processing
functions, or SkLearn functions,a nd the important bits are reproduced in this writeup. Of course, one can also 
follow the code to find the code that is actually executed. 

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

##### A note Concerning Subsequent Runs

The `detect_vehicles.py` file contains facilities for also reading a previously pickled classifier object from disk. 
This is useful e.g. if one changes some parameters of the pipeline that don't affect feature extraction, 
or wants to process another video. Once a classifier has been saved, one can run the detection again by issuing
 ```
detect_vehicles(previous_classifier_path="some_path")
 ```
 
By default, the processing pipeline saves the classifier in a default location, and this can be accessed by running the 
 function call above with the parameter `_DEFAULT_CLASSIFIER_PATH`, defined also in `detect_vehicles.py`  



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

I did not have a chance to play around with training the classifier much, especially as some very early attempts
seemed to produce reasonably good results. I just use the `LinearSVC` classifier of `scikit-learn` with its default 
parameters. So, after collecting the training data, training the classifier was just a matter of running the following
two lines. 

```
svc = LinearSVC()
svc.fit(features_train, labels_train)
```

This code is located at `classify/svm_classifier.py` file along with some processing related to getting the 
training data ready for processing. 

Using the default approach, the classifier is subsequently evaluated against a separate validation set. Using my 
training data, the result of running 

```
pred = svc.predict(features_valid)
score = accuracy_score(labels_valid, pred)
print("Successfully trained the classifier. Accuracy on test set was {}".format(score))
```

was typically along the lines of 
```
FIXME
```

so I figured the classifier as such was robust enough. 

#### The moving window search, tracking vehicles and removing false positives

For the labs preceding the projects, I had written a version of the function that lists all the 
candidate windows at a specific scale. The algorithm of mine differs slightly from the one provided 
as an example solution. The code is reproduced below. 

```python
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    x_start_stop[0] = 0 if x_start_stop[0] is None else x_start_stop[0]
    x_start_stop[1] = img.shape[1] if x_start_stop[1] is None else x_start_stop[1]
    y_start_stop[0] = 0 if y_start_stop[0] is None else y_start_stop[0]
    y_start_stop[1] = img.shape[0] if y_start_stop[1] is None else y_start_stop[1]
    overlap_coeffs = tuple([x[0] - x[1] for x in zip((1,)*2, xy_overlap)])

    step_x, step_y = tuple(np.product(x) for x in zip(xy_window, overlap_coeffs))

    y_win_top = y_start_stop[0]
    y_win_bottom = y_win_top + xy_window[1]

    window_list = []
    while True:
        x_win_left = x_start_stop[0]
        x_win_right = x_win_left + xy_window[0]
        if y_win_bottom > y_start_stop[1]:
            break
        while True:
            if x_win_right > x_start_stop[1]:
                break

            window_list.append(((x_win_left, y_win_top), (x_win_right, y_win_bottom)))

            # Increment the left and right positions of the window by step size in x direction
            x_win_left, x_win_right = tuple([int(sum(x)) for x in zip((x_win_left, x_win_right), (step_x, ) * 2)])

        # Increment the top and bottom positions of the window by step size in y direction
        y_win_top, y_win_bottom = tuple([int(sum(x)) for x in zip((y_win_top, y_win_bottom), (step_y, ) * 2)])
    return window_list

```
However, in my submission this code is not actually used as it is replace by the `find_cars` function 
included in the project instructions. 


#### Tracking & False Positives

To stabilize the vehicle lookup a bit, I rolled out a method of my own. To summarize, the method consists of 
keeping track of smoothed heatmaps and averaging them over previous frames before continuing to labeling and 
bounding box drawing. 

The tracking is performed using an instance of the class AggregatedHeatmap, found in `models/aggregated_heatmap.py`. 
An instance of this classed is kept in memory while processing the images, and the previous 5 smoothed 
heatmaps are kept as the object's member variables, updating them every time a new frame is processed. 

The actual heatmap used for labeling, then, a heatmap averaged along the time dimension from the individual smoothed 
heatmaps. This is done with the following aims: 

* I hope the smoothing to stabilize the heatmap across frames in areas where there are borderline detections.  
  Especially if there are gaps of a few pixels, I hope the smoothing to result in apparent "detections" also in those 
  areas. Conversely, in areas where there are only a few hot pixels, I hope to smooth them out so they don't 
  trigger a vehicle detection as easily as they would without this moothing. 
* The averaging hopefully results in a little bit more stability in portions of the video where the vehicles 
  are easily on-and-off detected  - that is, the detection is lost in some frames while in subsequent frames there is 
  a match again. 
* With the previous point in mind, spurious detections can hopefully now be more easily ignored as a one-frame 
  detection will be flattened out in the averaging operation.  
  
The `AggregatedHeatmap` class is reproduced entirely below: 

```python
SMOOTHING_KERNEL = np.array([[0.25, 0.25], [0.25, 0.25]])

class AggregatedHeatmap:
    def __init__(self):
        self.smoothed_heatmaps = np.zeros((5, 720, 1280))

    def process_new_heatmap(self, heatmap):
        self.smoothed_heatmaps = np.roll(self.smoothed_heatmaps, 1, 0)
        self.smoothed_heatmaps[0] = self.smooth_heatmap(heatmap)


    @staticmethod
    def smooth_heatmap(heatmap):
        return cv2.filter2D(heatmap, -1, SMOOTHING_KERNEL)

    def smoothed_heatmap(self):
        return np.average(self.smoothed_heatmaps, 0, [30, 28, 14, 10, 8])
```

And the part where it is used is in the heatmap processing (`mask/heatmap.py`): 

```python
heatmap = np.clip(heat, 0, 255)
aggregated_heatmap.process_new_heatmap(heatmap)
heatmap = _apply_threshold(aggregated_heatmap.smoothed_heatmap(), 8)
labels = label(heatmap)
```

Finally, it should be noted that I experimented also with a method where in the averaged heatmap I gave a 
multipyling weight factor to pixels where a detection was made across multiple frames. This is partly taken care
of by the averaging operation. Still, considering the arrays below, I wanted to experiment if boosting the one below 
would be beneficial in terms of detection performance. My findings are at this point inconclusive 
and I left this weighting scheme out of the submission as it added a lot of computational burden and the processing 
became way too slow. Anyhow, coming up with a weighting scheme was as simple as first calculating the matrix
```python
np.apply_along_axis(np.count_nonzero, 0, self.smoothed_heatmaps)
```
and then multiplying the averaged counts with this matrix element-wise. 

 
### The solution 

TODO: maybe  upload the video to YouTube and include a (image) link to the video in the writeup, as 
per https://stackoverflow.com/a/16079387



### Discussion
