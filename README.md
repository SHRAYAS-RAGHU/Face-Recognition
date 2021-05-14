# FACE-RECOGNITION

The face recognition library of python is built with inbuilt trained HOG and CNN (Resnet) Model that can identify human faces.

**Face Encodings:**
        
We can pass a human face image to the functions of the  library to
   * find out Facial encodings i.e. features like eyes, nose, mouth, eyebrows, etc that depicts a human face.
   * find out the location of the face in the image and return the co-ordinate list which can be used for drawing a bounding box.
   * find out the percentage of match between 2 faces using the encoded values.
There are many more functions we can explore.


**Face Classification:**

If we create a dataset of human faces with their name as Filenames, Face_recognition library can be used to compare the real - time captured face to a present dataset to find out the person.

**OUTPUTS :**
<p align = "center">
<img src = "https://user-images.githubusercontent.com/72727518/118289200-b82d6480-b4f2-11eb-953d-5b8caa5f5567.png" width = "400" height = "400">
<img src = "https://user-images.githubusercontent.com/72727518/118289241-c2e7f980-b4f2-11eb-91dd-72ae6fd87bad.png" width = "400" height = "400">
</p>

**USAGE :**

Install face-recognition library and change the path of training dataset to your storage directory.

**Reference** : https://pypi.org/project/face-recognition/
