import numpy as np
import cv2
import os
import face_recognition

folder = 'C:\\Users\\krshr\\Desktop\\Files\\Deep_learning\\Face_Recognition\dataset'
images = []
class_names = []
mylist = os.listdir(folder)

if '.ipynb_checkpoints' in mylist:
  mylist.remove('.ipynb_checkpoints')            # To strip out any additional files other than images

for cl in mylist:
    curr_img = cv2.imread(folder + '\\' + cl)
    images.append(curr_img)                      # Loading the images and
    class_names.append(os.path.splitext(cl)[0])  # their respective class name (Person name) from the filename

def find_enc(images):
    enc_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(img)[0]    # Function to find the face encodings. Encoding are facial features like eyes nose eye-brows.
        enc_list.append(enc)                                           
    return enc_list

known_end_list = find_enc(images)                  # Finding the face encodings for all the images in the dataset
print('Encoding complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgs = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    face_curr_frame = face_recognition.face_locations(imgs)                  # if face recognised sends the coordinate of Bounding-Box
    face_encodings = face_recognition.face_encodings(imgs, face_curr_frame)  # returns face encodings of the web cam image

    for encodeframe, facelocframe in zip(face_encodings, face_curr_frame):
        matches = face_recognition.compare_faces(known_end_list, encodeframe) # Compares known encodings with original encodings to find which of the faces does webcam face macth 
        facedis = face_recognition.face_distance(known_end_list, encodeframe) # Level of matching between current encoded values from webcam and dataset provided. Lowerthe value higher the match is.

        matchindex = np.argmin(facedis)

        if matches[matchindex]:
            name = class_names[matchindex].upper()
            y1, x2, y2, x1 = facelocframe
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            #cv2.rectangle(img, (x1, y2-35), (x2,y2),(0,255,0),cv2.FILLED) # uncomment if u ant the name to be in a color filled box
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
    cv2.imshow('web_cam', img)
    cv2.waitKey(0)


