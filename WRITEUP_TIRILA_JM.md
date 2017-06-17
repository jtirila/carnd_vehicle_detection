# Vehicle detection project, P5

**Writeup by J-M Tirilä**

(Work in progress.)

[//]: # (Image References)

[training_sample_img]: ./output_images/training_sample_img.png "An example training image"
[per_sample_y_channel_compaison]: ./output_images/fixme "The effect of per-image Y channel normalization"
[stacked_y_channel_comparison]: ./output_images/fixme "The effect of Y channel normalization on the stack of images"
[image_to_scan]: ./output_images/image_to_scan.png "The image to scan"
[candidate_image]: ./output_images/candidate_image.png "Candidate image"
[candidate_image_ch0]: ./output_images/candidate_image_channel_0.png "YCrCb conversion, first channel"
[candidate_image_ch1]: ./output_images/candidate_image_channel_1.png "YCrCb conversion, second channel"
[candidate_image_ch2]: ./output_images/candidate_image_channel_2.png "YCrCb conversion, third channel"
[image_to_scan]: ./output_images/image_to_scan.png "Another candidate image"
[another_candidate_image]: ./output_images/another_candidate_image.png "Another candidate image"
[another_candidate_image_features]: ./output_images/another_candidate_image_features.png "Another candidate image, features"
[yet_another_candidate_image]: ./output_images/yet_another_candidate_image.png "Another candidate image"
[yet_another_candidate_image_spatial_features]: ./output_images/yet_another_candidate_image_spatial_features.png "Another candidate image, spatial features"
[yet_another_candidate_image_2]: ./output_images/yet_another_candidate_image_2.png "Another candidate image"
[yet_another_candidate_image_2_hist_features]: ./output_images/yet_another_candidate_image_2_hist_features.png "Another candidate image, hist features"
[augmentation_example_images]: ./output_images/some_augmentation_images.png "Some example images from the augmenting set"
[car_training_img]: ./output_images/car_training_image.png "Traffic Sign 1"
[car_training_img_hog_ch0]: ./output_images/car_training_image_hog_ch0.png "Traffic Sign 2"
[raw_detection]: ./output_images/raw_detection.png "All detected boxes (for a single scale)"
[training_preprocessing_average]: ./output_images/training_preprocessing_average.png "An average preprocessing result"
[training_preprocessing_improved]: ./output_images/training_preprocessing_improved.png "A preprocessing result where batch seems to help in comparison to raw image"
[training_preprocessing_batch_advantage]: ./output_images/training_preprocessing_batch_advantage.png "A preprocessing result where batch seems to help in comparison to single image preprocessing"


## Notes on the submission

### Repository structure

For this project, I decided to use a conventional Python project hierarchy, with packages and modules arranged 
into a tree of directories and source code files grouped together according to related functionality. The 
structure of the project is a follows (omitting parts not essential fot this illustration): 

#### Hierarchy of packages and modules relevant to the submission

Note: the repository contains also files that are unused in the code for this submission. As of now, the ones listed 
below actually participate in processing the video.
```
|── README.md
│── detect_vehicles.py
├── WRITEUP_TIRILA_JM.md
├── carnd_vehicle_detection
│   ├── __init__.py
│   ├── classify
│   │   └── svm_classifier.py
│   ├── mask 
│   │   └── heatmap.py
│   ├── models
│   │   └── aggregated_heatmap.py
│   ├── preprocess
│   │   ├── __init__.py
│   │   ├── bin_spatial.py
│   │   ├── color_convert.py
│   │   ├── color_histogram.py
│   │   ├── extract_features.py
│   │   ├── hog_transform.py
│   │   └── read_classification_training_data.py
│   ├── traverse_image 
│   │   ├── __init__.py
│   │   ├── bin_spatial.py
│   │   ├── find_cars.py
├── images
│   ├── vehicles 
│   └── nonvehicles 
├── output_images
├── project_video.mp4
└── unit_tests
```

The top-level `carnd_vehicle_detection` directory acts as a container for everything, including documentation 
(readme, writeup). The image and video assets needed to run the code are **not** included in the repository. 
A subdirectory also named `carnd_vehicle_detection` contains just the Python code. 

The most important files the reviewers should pay attention to are `detect_vehicles.py` and 
`traverse_image/find_cars.py`. The rest mainly implement thin wrappers around OpenCV image processing
functions, or SkLearn functions,a nd the important bits are reproduced in this writeup. They are listed 
above anyhow. 

**When mentioning code file paths in this writeup, the root of reference is the innermost 
`carnd_vehicle_detection` directory by default. Some references are made to file outside the code 
directory and this is explicitly mentioned in those cases.**

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


## The project 

### Goals

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a 
  classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, 
  to your HOG feature vector. 
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream  
  and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles
* Estimate a bounding box for vehicles detected.

### My approach

In addition to the hints given in the project instructions, I decided to try adding some colorspace 
preprocessing, augment the training data set with some generated images, and implement a 
custom method for stabilizing the vehicle detections across frames.  

### Overview of my Solution in Terms of the Rubric Points 

#### Data Preprocessing

##### Image Transformation Using Histogram Equalization

Both when training the classifier and when making predictions, before entering the actual feature extraction loop, 
I first perform some simple preprocessing using my pet technique: histogram normalization of the Y channel after first 
converting the images to `YUV` color space, and then converting them back to `RGB`. This method is performed with 
the aim of enhancing contrast in the images. Below is an example image after this method applied.  

![image_to_scan][image_to_scan]

I also experimented with other techniques such as normalizing the saturation channel after a HLS transformation, 
and also some adaptive smoothing methods to get rid of noise in the images. However, especially the smoothing seemed 
like a suboptimal strategy as our images are rather low in pixel resolution and smoothing further blurs the 
valuable car pixels. I hence decided to go forward using just Y channel normalization. 

**An important note concerning training data normalization**

After a long period of trial and error, I realized something what was crucial to coming up with a functional pipeline: 
 
_Even though the `Y` channel normalization is a functional and useful technique, it must be properly applied
especially when the images in question only span a small area of the real world, and hence also a small range 
of luminosity (and other) values. Applied on individual small images (such as my training set), the results are 
suboptimal. The twist that helped me greatly improve the detection results was the the application of this 
normalization technique on an image formed by concatenating a greater batch of the training images together
so as to have more variability in the normalized image. I cannot emphasize enough how important this simple
finding was for the performance of my algorithm._

To be specific, before equalizing the `Y` channel histogram, I concatenated **all** of my training images into one 
large horizontal picture, and then after the normalization, sliced this huge image back into individual training
images. 

To demonstrate the effect, here are a few triplets with the original training image on the left, single-image 
`Y` channel histogram normalized image in the middle, and the batch normalized image on the right. In all of 
these images, the batch preprocessed image arguably outperforms the per-image preprocessed version, 
and at least in the last one some improvement can also be seen in terms of contrast, compared to the original 
image. 

![A normal preprocessing result][training_preprocessing_average] 
![Batch preprocessing has an edge over per-image preprocessing][training_preprocessing_batch_advantage]
![Preprocessed slightly better than original][training_preprocessing_improved]

##### Colorspace Conversion

Even though the color space conversion introduced in the labs preceding the project is performed as part of the feature 
extraction pipeline, I think it lies conceptually more in the preprocessing domain. My experiments with color spaces
indicated that `YCrCrCb` was a strong candidate and that is what  I ended up using. Below are again (luminosity 
normalized) original image and the individual color channels after a `YCrCb` conversion: 

![candidate_image][candidate_image]
![candidate_image_ch0][candidate_image_ch0]
![candidate_image_ch1][candidate_image_ch1]
![candidate_image_ch2][candidate_image_ch2]


##### Augmenting the Data with Examples From the Project Video 

In addition to the training files provided for the project, I also decided to augment the data set by 
taking screenshots of the project video, multiplying them adding random rotations and scalings and then 
adding these to the training data set. 

The code that produces the augmentations was a one-time script. It can be found at 
`script/one_off_scripts/augment_data.py`. There is lot of routine filename changing included (to get rid of
spaces in filenames) but the most important part that introduced the rotations and scalings is included below.  

```python
def save_rotated_scaled_versions(filenames, images, rounds):
    for round in range(rounds):

        rotation_angles = [18 * (random() - 0.5) for _ in range(len(images))]
        scale_coeffs = [1.1 + 0.2 * random() for _ in range(len(images))]

        for ind, img_and_filename in enumerate(zip(images, filenames)):
            img, filename = img_and_filename
            print("round {}, ind {}, filename {}".format(round, ind, filename))
            matr = cv2.getRotationMatrix2D((32, 32), rotation_angles[ind], scale_coeffs[ind])
            new_image = cv2.warpAffine(img, matr, (64, 64))
            new_image_filename = index_filename(filename, "round_{}_image_{}".format(round, ind))
            # print(new_image_filename)
            mpimg.imsave(new_image_filename, new_image[:, :, :3])

```

##### Color Space

As part of the pipeline, the images can be converted into another color space before further feature 
extraction. To simplify the target colorspace perameters, I wrote a wrapper script for the OpenCV function: 

```python
def convert_color(img, conv="RBG"):
    """Convert color to the specified color space.
    
    :param img: The original image, assumed to be in RGB
    :param conv: A string containing the new colorspace name, accepted strings are HSV, HLS; YCrCb, FIXME what else
    
    :return: the transformed image"""
    if conv != 'RGB':
        color_specifier = "COLOR_RGB2{}".format(conv.upper())
        feature_image = cv2.cvtColor(img, getattr(cv2, color_specifier))
    else:
        feature_image = np.copy(img)
    return feature_image
```

This way, I can just specify e.g. `YCrCb` or `HLS` and do not neeed to remember the full CV2 colospace specifiers. 
The choice of color space was already discussed above in the preprocessing section. 


#### Feature extraction

Feature extraction is performed using the methods introduced in the labs preceding the project.  

##### Spatial features

The spatial features vector is just a compressed version of the original image. While the main characteristics of e.g.  
a vehicle are still preserved in a, say, 32x32 pixel image, I doubt the usefulness of such a feature vector
depends heavily on the type of classifier used. As I ended up using an SVM classifier for the project,  
I am not really sure if any linear decision surface would be able to use the compressed version of an 
image very efficiently.

If I were to use a convolutional neural network, for example, the situation might be different, though. 

Below is the code used for extracting the spatial features.  

```python
def bin_spatial(img, size=(32, 32)):
    """Just resize an image and return the color values as a 1-dimensional vector
    
    :param img: the original image
    :param size: A two-tuple containing the new size
    :return: a 1-dim feature vector"""

    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))
```

Below The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.is an example of another candidate image together with its spatial feature vector 
(32x32x3 = 3072 pixels, using the `YCrCb` color space). 

![yet_another_candidate_image][yet_another_candidate_image]
![yet_another_candidate_image_spatial_features][yet_another_candidate_image_spatial_features]

##### Color Histogram 

As for the color histogram, I think its usefulness in a SVM classifier context may be a bit better justified 
than that of the spatial feature vector. 

Color histogram nicely captures the relative count of different color value ranges in an image, on a 
per-channel basis. The value range is divided into bins of equal size each bin is subsequently represented 
by how many of all the color values in the input fall into the bin. 

This is especially true for alternative color spaces that 
are probably better able to distinguish between saturated "artificial" colors used in vehicles, and 
the more washed out background of a road environment. Care should be taken though so the classifier 
does not go totally wild in a colorful city environment. 

Even for the color histograms, my findings did not suggest an immediate usefulness, but in a real-world 
detection problem, I would probably still consider including them due to the potential effects mentioned above. 

However, in conclusion regarding both the spatial and color histogram features, I had a hard time obtaining a stable 
vehicle detection so I ended up including all of the suggested features
anyway, hoping they would provide even a tiny bit of assistance in detecting vehicles in more challenging 
portions of the video.  

The color histogram feature vector used in this project is computed using the following code. 

```python
def color_hist(image, nbins=32, bins_range=(0, 256)):
    """Compute the color histogram of a 3-channel image
    
    :param image: the original image
    :param nbins: The number of histogram bins to divide the color range into
    :param bins_range: The range of values
    
    :returns: The histogram values concatenated into a feature vector"""

    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(image[:,:,0], bins=nbins)
    channel2_hist = np.histogram(image[:,:,1], bins=nbins)
    channel3_hist = np.histogram(image[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features
```

Below is first yet another candidate image together with its histogram feature vector 
(using 16 bins per channel and the `YCrCb` color space).

![yet_another_candidate_image_2][yet_another_candidate_image_2]
![yet_another_candidate_image_2_hist_features][yet_another_candidate_image_2_hist_features]

##### The HOG features

The histogram of oriented gradients is a nice approach to indclude some generic shape templates in the feature vector.  
The method determines (discrete) gradient directions an magnitudes for each pixel in the image, and then bins them 
according to the chosen cell size and blocking scheme. 

The result is a histogram of gradient directions for each cell, 
weighted by the magnitudes, so that the dominant gradient directions are extracted. 
This is illustrated in the image below, with the first figure containing the original image 
and the second one its HOG visualization (first of the three color channels). 

![car_training_img][car_training_img]
![car_training_img_hog_ch0][car_training_img_hog_ch0]

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

Below is an example of first the original candidate image (corresponding to a viewing window) and subsequently 
its normalized feature vector extracted using only the HOG extractor for all channels in an 
`YCrCb` image.

![another_candidate_image][another_candidate_image]
![another_candidate_image_features][another_candidate_image_features]


**A note about color space**: the project rubric requires that at this points, colorspace conversion is also discussed. 
I included the discussion in my section on preprocessing, please refer to that section above for more information. 

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

###### The Sliding Window Modifications to the Provided Example Code

So, as mentioned above, I basically adopted the windowing scheme provided as example code. I made a
couple of changes, though: 

**Cells per step**

For window overlap, I figured at larger scales it may be beneficial to not take too wide steps so 
I set the `cells_per_step` variable that controls overlap to just 1 for larger windows as follows:
  
```python
if scale < 1.9:
    cells_per_step = 2 # Instead of overlap, define how many cells to step
else:
    cells_per_step = 1
```  

The rationale for this choice is that I figured it is more imprtant to detect vehicles close to the camera, 
and those vehicles would typically be bigger in scale than ones further away. 

**Area of interest also in X direction**

The template only cropped the image in the vertical direction. I made the changes needed to the code to also 
be able to restrict the are scanned horizontally. This is done on a per-scale basis so that for larger scales, 
the whole width of an image is scanned, while for smaller scales, matches are only searched closer to the center 
of the image. This was done to prevent spurious matches outside of the lane of interest. 

##### Normalizing the feature vectors. 

For the feature vector normalization, I used the `StandardScaler` class of `sklearn`. The code below illustrates 
how it is used for the training features. The usage for other purposes (testing of candidate images, validation) is 
done similarly, just omitting the first two lines.

```python
scaler = StandardScaler()
scaler.fit(extracted_features_train)
scaled_features_train = scaler.transform(extracted_features_train)

```


#### Training the classifier

I briefly tried experimenting with different classifiers, as the 
[sklean comparison charts](http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) 
indicated some
methods other than the LinearSVM could yield more complex decision boundaries. I tried to use a RandomForestClassifier,
but the results were not improved at least in my trials so I ended up just using the  `LinearSVC` classifier 
of `scikit-learn` with its default parameters. So, after collecting the training data, training the classifier was 
just a matter of running the following
two lines. 

```
svc = LinearSVC()
svc.fit(features_train, labels_train)
```

This code is located at `classify/svm_classifier.py` file along with some processing related to getting the 
training data ready for processing. 

Using the default approach, the classifier is subsequently evaluated against a separate validation set. Using my 
training data, as result of running 

```
pred = svc.predict(features_valid)
score = accuracy_score(labels_valid, pred)
print("Successfully trained the classifier. Accuracy on test set was {}".format(score))
```

the accuracy was typically 98-99 percent for larger feature sets and 93-94% if a shorter feature vector
 was used (e.g. only one channel of the HOG features).

I figured with something around 98-99% the classifier as such was robust enough so I decided to go 
forth with this solution. 

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
SMOOTHING_KERNEL = np.ones((8, 8))
SMOOTHING_KERNEL[2:-2, 2:-2] = 2
SMOOTHING_KERNEL[3:-3, 3:-3] = 4
SMOOTHING_KERNEL = SMOOTHING_KERNEL / np.sum(SMOOTHING_KERNEL)

class AggregatedHeatmap:
    def __init__(self):
        self.smoothed_heatmaps = np.zeros((7, 720, 1280))

    def process_new_heatmap(self, heatmap):
        self.smoothed_heatmaps = np.roll(self.smoothed_heatmaps, 1, 0)
        self.smoothed_heatmaps[0] = self.smooth_heatmap(heatmap)


    @staticmethod
    def smooth_heatmap(heatmap):
        return cv2.filter2D(heatmap, -1, SMOOTHING_KERNEL)

    def smoothed_heatmap(self):
        return np.average(self.smoothed_heatmaps, 0, [50, 40, 30, 20, 20, 10, 10])  \
               * np.count_nonzero(self.smoothed_heatmaps, 0)
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
    def smoothed_heatmap(self):
        return np.average(self.smoothed_heatmaps, 0, [30, 28, 14, 10, 8]) \
            * np.apply_along_axis(np.count_nonzero, 0, self.smoothed_heatmaps)
```
and then multiplying the averaged counts with this matrix element-wise. 

This stabilizing tehcnique turned out to indeed improve the tracking. However, it came with a cost: the computational 
demands of the algorithm were increased quite much. 

I decided to go forward with this approach, however, to further investigate this idea of heatmap aggretation.  

Below are images of a plain heatmap, then an aggregated heatmap produced with the method above. 

### The solution 

#### Training the Model  

The orchestration of processing the individual frames and saving them into a new clip is taken care of in the 
`detect_vehicles.py` file, using `MoviePy`'s `fl_image` function. In the current form, the processing takes 
 two and a half hours, so to use this method in any real world settings, performance increasements would be 
 mandatory. I suspect there is a way of performing the numpy array averaging and weighting by number of nonzero 
 elements on an axis in a single pass. 
 
 #### The result
 
 I uploaded the result video to YouTube. It is embedded below if you are reading a rendered version of this writeup. 
 
[My output video](https://img.youtube.com/vi/wW3-68DhNfI/0.jpg)](https://www.youtube.com/watch?v=wW3-68DhNfI) 

### Discussion

I am not sure if I failed to really nail the parameters or is something else was a bit off, but in any case, 
I found the project quite challenging. So I am not quite happy with the final submission as it has probably at least 
the following shortcomings. 

* The pipeline is way too slow. Using the detection count enhanced aggregated heatmap, It takes several seconds in my 
  laptop to process each frame. This is somewhat mitigated by using the simpler version of the aggregate heatmap method, 
  as well as reducing the number of scales used for scanning or images to scan. Then again, this also leads to 
  decreased detection performance. 
* I also suspect that the detection behavior (parameters thresholds etc.) are now overfit for this particular 
  video and would perform poorly if e.g. the lightness of the video were much different. 
  
However, I think there are also things I am happy with, or that were at least steps in the right direction:  

* The method of stacking the training images together for `Y` channel normalization. This had a dramatic 
  positive effect on detection performance. 
* I also think the heatmap normalization method I ended up using has some potential to it. It is not perfect for 
  this submission and it may lack some major insights that would make such processing useful, but anyhow 
  I think this method of weight-averaged heatmaps has great potential to emphasize "long-term detections" and 
  discard spurious false positives without too much code or involved logic. This is also something I would like to 
  investigate further, also from a mathematical point of view. 
* Even though I had to rush with the sumission and the repository structure is hence not properly finished, I think 
  the modularized structure (combined with revised pieces from my previous projects) would make a decent start 
  for a bit more general video processing toolset. I would be great to have time to really structure everything 
  nicely and have everything generic enough to be able to perform various kinds of video and image processing 
  and feature extraction tasks from a single repository using more or less plug and play methods. 

Things I would have liked to investigate but did not have time for: 

* This project sparked my interest in keeping track of vehicles over frames, and from a more generic perspective, 
  of following the paths of different objects over time and observing / detecting how these objects may seemingly 
  merge (when overlapping) and, after merges, how they can be identified again. Specifically, even though 
  out of scope of this particular project, being able to track, say, a particular object of interest in situations
  where it may overlap with similar objects would be an interesting dilemma to research. 
* In a similar fashion this made me think of lane line "detections" (or rather predictions) when they are partly 
  occluded by vehicles or other objects. There is an application of this I would also like to look into:  
  - The vehicle search would be much more efficient if one had a reasonable estimate of where the road edges are. 
    However, this becomes kind of a chicken-and-egg situation when vehicles occlude parts of the edges and hence 
    prevent reliable edge detection.
* Also, I hope at some point to be able to investigate any performance improvements that might be possible with this
  algorithm. There is quite a bit of Numpy array manipulations going on, and I suspect at least moderate improvements
  could be achieved by just being more careful not to introduce superfluous matrices. Also, there may be 
  room in the code to perform more of processing in a single pass. I suspect the feature extraction part of the code 
  could all be quite easily done for the whole of the area of interest simultaneously, and then for the 
  candidate "vehicle" windows, one could then just pick part of the precomputed feature matrix. 
* Another rather easy   
