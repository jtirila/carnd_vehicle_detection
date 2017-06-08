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
