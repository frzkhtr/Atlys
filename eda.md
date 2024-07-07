### Exploratory Data Analysis

Mobile Identity Document Video dataset is an open-source dataset containing 500 video clips for 50 different identity document (passport, liscence etc)

I used python module [midv500](https://github.com/fcakyon/midv500/tree/master) to download and unzip dataset.


Total Size of the dataset is around 47 Gb.

#### Directory & Data Structure

The dataset is downloaded in working directory. Full directory structure is in directory_structre.json. 
In short, Data is first seggerated in folders based on region-id. Inside each region-Id folder. It is seperated based on data type, having 3 subfolders:
    - images: contains images(.tiff)
    - videos: contains (.mp4)
    - ground-truth (.json)


Inside these subfolders, there are many subfolders which contains their respective data. It will follow same name structure for images, videos and ground_truth folders.


#### Images:
There are 15000 images, each region-id folders having 300 images. The images further evenly distributed images subfolders each having 30 images. 

Image Resolution Statistics:
  Mean Image Resolution: [1920. 1080.]
  Minimum Image Resolution: [1920 1080]
  Maximum Image Resolution: [1920 1080]


#### Ground Truth:
gound_truth folders contain the bounding box of images of same name-structure. It is in json format, containing the quad point location in images which identify the document/id within the images.

