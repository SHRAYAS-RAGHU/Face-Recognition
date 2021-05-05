import numpy as np
import cv2
import os
import face_recognition

folder = r'C:\Users\krshr\Desktop\Files\Deep_learning\Face_Recognition\dataset'
images = []
class_names = []
mylist = os.listdir(folder)
if '.ipynb_checkpoints' in mylist:
  mylist.remove('.ipynb_checkpoints')

for cl in mylist:
    curr_img = cv2.imread(folder + '\\' + cl)
    images.append(curr_img)
    class_names.append(os.path.splitext(cl)[0])

def find_enc(images):
    enc_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(img)[0]
        enc_list.append(enc)
    return enc_list

known_end_list = find_enc(images)
print('Encoding complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgs = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    face_curr_frame = face_recognition.face_locations(imgs)
    face_encodings = face_recognition.face_encodings(imgs, face_curr_frame)
    
    """
    FACE LOCATIONS:

    Returns an array of bounding boxes of human faces in a image

    :param img: An image (as a numpy array)
    :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
    :param model: Which face detection model to use. "hog" is less accurate but faster on CPUs. "cnn" is a more accurate
                deep-learning model which is GPU/CUDA accelerated (if available). The default is "hog".
    :return: A list of tuples of found face locations in css (top, right, bottom, left) order

    FACE ENCODINGS:
    Given an image, return the 128-dimension face encoding for each face in the image.

    :param face_image: The image that contains one or more faces
    :param known_face_locations: Optional - the bounding boxes of each face if you already know them.
    :param num_jitters: How many times to re-sample the face when calculating encoding. Higher is more accurate, but slower (i.e. 100 is 100x slower)
    :param model: Optional - which model to use. "large" (default) or "small" which only returns 5 points but is faster.
    :return: A list of 128-dimensional face encodings (one for each face in the image)
    """

    for encodeframe, facelocframe in zip(face_encodings, face_curr_frame):
        matches = face_recognition.compare_faces(known_end_list, encodeframe)
        facedis = face_recognition.face_distance(known_end_list, encodeframe)
        print(matches, facedis)
        matchindex = np.argmin(facedis)

        if matches[matchindex]:
            name = class_names[matchindex].upper()
            y1, x2, y2, x1 = facelocframe
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
    cv2.imshow('web_cam', img)
    cv2.waitKey(0)


