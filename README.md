# Vehicle detection

## General info

This repository will contain all my work for the Udacity CarND Term1 final project.

As of now (7 June 2017), this is heavily work in progress but I hope to get something reasonable running in about 
a week. 

Note on code copyright: Even though the overall structure and also much of the details will be written by me, much 
of the credit goes to the Udacity team. I'm not aware of any licensing aspects, please contact me if you need some
info on this.

## The project

The instructions for the vehicle detection project are part of the CarND program and are not reproduced here. 
The general framwork may be obvious from the writeout that will be included in this repo once the project is ready, but 
I can give no guarantee on whether the code or the writeup will be useful to anyone outside of the CardND context. 

## The structure of the code and running the pipeline

### Prerequisities

* Some Python 3.x version. The project has been written on Python 3.5. I am not aware if compatibility varies between 
  different minor versions of Python 3. Possibly package dependencies could cause difficulties with some combinations.  
* All the Python packages needed to run the code, including Numpy, OpenCV and MatPlotLib. 
  The easisest way to get both Python and the packages up and running is probably to install the 
  [Udacity started kit](https://github.com/udacity/CarND-Term1-Starter-Kit). 
* A suitable set of training images to train the vehicle detection classifier. These files will not be included in the 
  repository. The images have to be placed in the `images` subdirectory of the top level of the 
  project, with vehicle images place in subdirectories of `vehicle` and non-vehicle images in subdirectories of 
  `nonvehicle` within the image directory.
* The project video is also omitted from this repository. You should find 
  [the one used for the project](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/project_video.mp4) 
  or an equivalent video. Probably there are some requirements on the type of video but I don't know them as of now. 
  
  By default, there needs to be a video called `project_video.mp4` in the top level directory. 
  See `detect_vehicles.py` for how you can provide a different filename as a parameter. 
* You probably also need to have `ffmpeg` installed. Not sure about this. If the processing fails, you 
  can try installing `ffmpeg`. 

With these in mind, you can simply run the vehicle detector script by 
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

## The writeup

(Note: no point following the link just yet.)

Udacity people, look this way please. My written report can be found by following [this link](./WRITEUP_TIRILA_JM.md).

