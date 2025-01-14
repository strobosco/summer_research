# Aria Project 3D Facial Reconstruction

## Pittsburgh Experimental Research Group

This documentation will explain the code layout as well as some basic functionality required to run PERG's 3D Facial Reconstruction Tool. For a more in-depth look into the inner workings of the code please refer to the **`Aria Setup`** guide on the Notion page (ask Niccolo, Tao, or Professor Shangguan).

# Code Layout

```python
+-- archive // repository containing code for MyoWare sensors
|   +-- arm
|   +-- face
|   +-- sensor_test
|
+-- data_collection // directory containging the stored test data (automatically created)
|
+-- common.py // contains code required for Aria library
|
+-- mystreamer.py // file that connects to glasses, starts streaming
|
+-- myvisualizer.py // file containing observer and visualizer code
|
+-- requirements.txt // envirnment requirements
|
+-- *.ipynb // miscellaneous post-processsing files
```
