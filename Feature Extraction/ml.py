#################################################
#               Imported Libraries              #
#################################################
import math
from scipy.spatial import distance
import cv2
from mlxtend.image import extract_face_landmarks
import numpy as np
import os
import csv

#################################################
#               Feature Extraction              #
#################################################
def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[14], mouth[18])
    C = distance.euclidean(mouth[12], mouth[16])
    mar = (A ) / (C)
    return mar

def circularity(eye):
    A = distance.euclidean(eye[1], eye[4])
    radius  = A/2.0
    Area = math.pi * (radius ** 2)
    p = 0
    for i in range(5):
        p += distance.euclidean(eye[i],eye[i+1])
        if i == 4:
            p += distance.euclidean(eye[5], eye[0])
    return 4 * math.pi * Area /(p**2)

def mouth_over_eye(eye):
    ear = eye_aspect_ratio(eye)
    mar = mouth_aspect_ratio(eye)
    mouth_eye = mar/ear
    return mouth_eye

def head_drop(eye):
    A = distance.euclidean(eye[0], eye[35])
    B = distance.euclidean(eye[1], eye[34])
    C = distance.euclidean(eye[2], eye[33])
    return (A+B+C)/3

#################################################
#          Parameters for normalization         #
#################################################
def normalization_parameters(directory,video):

    # get first 3 images for this person
    i = 0
    first3 = []
    for filename in os.listdir(directory):  # run on all files in folder
        x0 = filename.split('.')
        vid = int(x0[0])
        if vid == video:    # normalize corresponding person's features
            img = cv2.imread(os.path.join(directory,filename))  # read image
            landmarks = extract_face_landmarks(img) # get facial feature's coordinates
            landmarks = np.array(landmarks)
            if landmarks.size > 1:   # face detected
                first3.append(os.path.join(directory, filename))
                i += 1
                if i == 3:
                    break
    # print(first3)
    
    # extract features from 3 first images  
    features = []
    for image in first3: # 3 images of the same person   
        img = cv2.imread(image) # read image
        landmarks = extract_face_landmarks(img) # get facial feature's coordinates
        data = np.array(landmarks)
        nose = data[32:68]
        eye = data[36:68]
        ear = eye_aspect_ratio(eye)
        mar = mouth_aspect_ratio(eye)
        cir = circularity(eye)
        mouth_eye = mouth_over_eye(eye)
        head = head_drop(nose)
        features.append([ear, mar, cir, mouth_eye,head])
    parameters = []    
    for i in range(5):  # add all 5 features to list
        parameters.append([features[0][i],features[1][i],features[2][i]])
    mean = []
    std = []
    for i in range(5):
        mean.append(np.mean(parameters[i])) # each feature's mean
        std.append(np.std(parameters[i]))   # each feature's standard deviation
    return mean,std

#################################################
#              Feature Normalization            #
#################################################
def normalize(directory,features,video):
    normalized = []
    mean,std = normalization_parameters(directory,video)
    for i in range(5):
        normalized.append(abs((features[0][i] - mean[i]))/ std[i])
    return normalized

directory = r'D:\Drv-Dev\Python\13- Final Project\Frames'
final_features = []

#################################################
#         Extract features and normalize        #
#################################################
for filename in os.listdir(directory):  # run on all files in folder
    checked = False             # boolean variable indicating if video has been checked before
    x0 = filename.split('.')
    video = x0[0]   # video number
    x1 = x0[1].split('-')
    frame = x1[1]   # frame number
    print(video,frame)

    # check if video has been seen before
    with open('features.csv') as file:  
        reader = csv.reader(file)
        for row in reader:
            if video in row[0]: 
                checked = True
                break
    
    # check video
    if not checked: 
        # print('checking video')
        img = cv2.imread(os.path.join(directory,filename))  # read image
        landmarks = extract_face_landmarks(img) # get facial feature's coordinates
        landmarks = np.array(landmarks)
        if landmarks.size > 1:   # face detected
            # print('found face')
            data = np.array(landmarks)
            features = []
            nose = data[32:68]
            eye = data[36:68]
            ear = eye_aspect_ratio(eye)
            mar = mouth_aspect_ratio(eye)
            cir = circularity(eye)
            mouth_eye = mouth_over_eye(eye)
            head = head_drop(nose)
            features.append([ear, mar, cir, mouth_eye,head])    # list of features
            ear_n,mar_n,cir_n,mouth_eye_n,head_n = normalize(directory,features,int(video))  # list of normalized features
            final_features.append([video,frame,features[0][0],features[0][1],features[0][2],features[0][3],features[0][4],ear_n,mar_n,cir_n,mouth_eye_n,head_n])    # full row of data
            # print('--- features ---')
            print(final_features[-1])
        else:
            print('no face detected')
    else:
        print('frame already seen')

##########################################################
#                   add all data to csv                  #
##########################################################
with open('features.csv','a',newline='') as f:  # add all data to csv
    thewriter = csv.writer(f)
    for example in final_features:
        thewriter.writerow(example)
