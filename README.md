# Localization of Transaprent Cups using Machine Learning

## Synopsis
This topic is taken from real-time applications in industries. If the bottles or cups moving
vertically in position in a conveyor, there are possibilities for bottles or cups to fall horizontally
on the conveyor. In those cases, the cup must be detected and placed in an upright position
using pick and place robot for the further process to be continued. The purpose of this project
is to detect, localize and classify the transparent cups using machine learning with TensorFlow.
The basic step in computer vision is object detection and classification. Detecting the
transparent object is complex because there will be some reflections due to light or some
external factor.

Yolo is the state of art approach in machine learning which is used to detect real-time objects.
In the basic YOLO approach, bounding boxes technique is used for object detection. The neural
network predicts key point pairs and class probabilities directly from a full image in only one
evaluation. In this project, the key point pair technique is used for detecting the object and
classify the full image. Key point pairs represent the spatial coordinates of the object. I used
Tiny YOLO v2 with GPU support which achieves up to 244 FPS. This architecture is remarkably
fast and objects detection performance-enhanced end-to-end directly.

<p align="center">
  <img src="https://github.com/saikrishnathonduru/Localization_of_Transaprent_Cups/blob/main/Images/ML.PNG?raw=true" alt="Sublime's custom image"/>
</p>

To detect the lying transparent cups, key point pairs of the cups are first needed. Then training
the model with key point pairs the detection parameters x, y, alpha and confidence of the
object can be obtained. Then the same approach is used to detect both lying and upright
transparent cups using a one-hot encoder for class probabilities (stating 0 for lying cup and 1
for upright cup). For the execution of this approach, the class was added in pre-processing,
network and loss function and detection of the class probabilities was accomplished.

## Approach
The following procedure is used to accomplish the objective.
1. Dataset creation of fallen and upright cups using manta G-032 B camera wherein image
pixel size should be 128 x 128.
2. Generation of a huge amount of data for good predictions
3. Installation of tensor flow GPU and run the program on it for fast computation
4. Modification of the yolo to detect the key point pairs
5. Evaluation of the custom loss for better predication rate
6. Detection of the cups and classify the lying and upright cups with class probabilities of
one hot encoder.
7. Classification of lying cups as red key point pair and upright cup as green key point pair.
So, the key point pair denotes the center point of the cup as well as the direction.
8. Detection of the mini metal shafts in a box and robot must pick and place where it is
needed.
9. Creation of a dataset using a manta G-032 B camera and training of the dataset.
10. Testing the new image for detection
11. Analyzing the statistical results and determine further steps to proceed for mini metal
shafts detection.

## Prerequisite’s modules 
The following modules are used in programming and discussed in short detail.
  ### Python
Python is one of the programming languages which is used for simplifying complex
applications by coding. Python 3.6.8 version is used in our system because it has all the
necessary libraries. This version introduced the standard type of annotations of function
parameters which adds syntax to Python for annotating the types of variables including
instance and a class variable.
  ### Tensorflow
Tensorflow is a free open source framework created and released by Google. It computes
extremely fast and line-by-line training is possible. It can be used to create deep learning
models directly. Tensorflow 1.14.0 is used in our system for fast computation. It has excellent
library management, quick updates and frequent new releases with new features. Debugging
is simple by executing the sub-parts of a code.
  ### Keras
Keras is an open-source neural network library which is written in python. It is an API that
makes it easy to learn and use. Implementations of ideas are easier and faster. Keras 2.3.1
version is used in our system as it is an updated version. There is no key scientific reason to
use this version and in my point of view, it maintains compatibility with TensorFlow.
  ### OpenCV
OpenCV means open-source Computer Vision. It is a library of programming functions which
mainly aimed at real-time computer vision. It is used for all the operations related to images.
OpenCV-python 4.1.2.30 version is used in this system.

## Validation of Transaparent Cups
### Detection of lying transparent cups
Here only the lying transparent cup is detected. After creating the image dataset, the model
is trained as per the procedure stated earlier. Eventually, it detects the cup at the end. Initially,
fewer images were used for training to check the functioning of the program as less image
dataset predictions were not good. The size of the images trained is 120 and the accuracy of
prediction is 40 %. When detecting only the lying transparent cups, there were no much
difficulties occurred. In the prediction, the output looks like x, y, alpha, and confidence. Due
to the single class, it is not included in the output. The figures below
are the predicted images of the lying transparent cups. Here a circle and a line are used for
the detection of the cup. The circle with a radius of 7 mm and a thickness of 1 mm which
locates the center of the object. The line with distance as per the coordinate dimensions takes
place and the thickness shows the direction of the transparent cup lying. The channel of colour
used was red to specify the lying cup. 


<p align="center">
  <img width="400" height="400" src="https://github.com/saikrishnathonduru/Localization_of_Transaprent_Cups/blob/main/Images/ir_0001_detected.bmp?raw=true" alt="Sublime's custom image"/>
</p>

<p align="center">
  <img src="https://github.com/saikrishnathonduru/Localization_of_Transaprent_Cups/blob/main/Images/Lying_cup1_Output.PNG?raw=true" alt="Sublime's custom image"/>
</p>

<p align="center">
  <img width="400" height="400" src="https://github.com/saikrishnathonduru/Localization_of_Transaprent_Cups/blob/main/Images/ir_0003_detected.bmp?raw=true" alt="Sublime's custom image"/>
</p>

<p align="center">
  <img src="https://github.com/saikrishnathonduru/Localization_of_Transaprent_Cups/blob/main/Images/Lying_cup2_Output.PNG?raw=true" alt="Sublime's custom image"/>
</p>

### Detection of upright transparent cups

<p align="center">
  <img width="400" height="400" src="https://github.com/saikrishnathonduru/Localization_of_Transaprent_Cups/blob/main/Images/Upright_Cup1.PNG?raw=true" alt="Sublime's custom image"/>
</p>

<p align="center">
  <img src="https://github.com/saikrishnathonduru/Localization_of_Transaprent_Cups/blob/main/Images/Upright_Cup1_Output.PNG?raw=true" alt="Sublime's custom image"/>
</p>

<p align="center">
  <img width="400" height="400" src="https://github.com/saikrishnathonduru/Localization_of_Transaprent_Cups/blob/main/Images/Upright_Cup2.PNG?raw=true" alt="Sublime's custom image"/>
</p>

<p align="center">
  <img src="https://github.com/saikrishnathonduru/Localization_of_Transaprent_Cups/blob/main/Images/Upright_Cup2_Output..PNG?raw=true" alt="Sublime's custom image"/>
</p>

### Validation of transparent cups

<p align="center">
  <img src="https://github.com/saikrishnathonduru/Localization_of_Transaprent_Cups/blob/main/Images/CupsTable.PNG?raw=true" alt="Sublime's custom image"/>
</p>

<p align="center">
  <img src="https://github.com/saikrishnathonduru/Localization_of_Transaprent_Cups/blob/main/Images/CupsGraph.PNG?raw=true" alt="Sublime's custom image"/>
</p>

The above table shows the observations and graph depicts the validation accuracy of the
transparent cups. The images from 640 to 6912 were trained by keeping constant epochs and
saving the respective weight files separately. Predicting the new image dataset and calculate
the accuracy by averaging the sample of 30 validated images. Let’s take a simple example, In
order to answer every question in examinations the preparation behind can be correlated with
the above training method. i.e If only a few topics (No. of Pages) are studied in the entire book
thoroughly (Epoch Count) then questions pertaining to that specifying topic could be
answered perfectly. On the other hand, if all topics ( More No. of the pages) are studied few
times ( Moderated Epoch count) eventually all questions can be answered that represent fast
reply or computation speed at moderate accuracy. Understanding is a new image dataset and
reply is computation speed. Figures shows clearly the accuracy increasing gradually by
increasing the number of the variance of images.

