##########################################################
#                     Imported Libraries                 #
##########################################################
import cv2
from scipy.spatial import distance
from imutils import face_utils, resize
import dlib
import math
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, make_scorer

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

def head_drop(nose):
    A = distance.euclidean(nose[0], nose[35])
    B = distance.euclidean(nose[1], nose[34])
    C = distance.euclidean(nose[2], nose[33])
    return (A+B+C)/3

#################################################
#                   Start Webcam                #
#################################################
print("-> Loading the predictor and detector...")
detector = dlib.get_frontal_face_detector() # face detector
# detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")    #Faster but less accurate
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')   # face predictor

print("-> Starting Video Stream")
vs = cv2.VideoCapture(0)    # start video stream

#################################################
#       Machine Learning Classification         #
#################################################
def classify(model,x_train,x_test,y_train,y_test):
    model.fit(x_train,y_train)  # train model
    y_pred =  model.predict(x_test) # get prediction on unseen examples
    confusion = confusion_matrix(y_test, y_pred)    # confusion matrix
    print(confusion[:2,:2])
    accuracy = accuracy_score(y_test,y_pred)    # compute model accuracy
    print('Accuracy: ' + str(accuracy))
    return confusion

df = pd.read_csv('D:/Drv-Dev/Python/13- Final Project/merge.csv')   # read training data
X = df.drop(columns=['Label','Video','Frame','EAR_N','MAR_N','CIR_N','ME_N','HEAD_N'])  # drop normalized features
# X = df.drop(columns=['Label','Video','Frame'])
Y = df['Label']
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42) # split to train and test
print('----------- Random Forest --------------')
# model = RandomForestClassifier()
model = RandomForestClassifier(ccp_alpha=0,criterion='entropy',max_depth=16,max_features='log2',max_samples=0.5,n_estimators=100,min_samples_split=20)

confusion = classify(model,x_train,x_test,y_train,y_test)
print('----- Feature Importance -----')
feature_importance = list(model.feature_importances_)   # feature importance
print(feature_importance)
print("5 Feature's importance: " + str(sum(feature_importance[:5])))  # importance of 5 features to the machine learning model
print("5 Normalized feature's importance: " + str(sum(feature_importance[5:])))  # importance of 5 normalized features to the machine learning model
# print('----------- Bagging --------------')
# model = BaggingClassifier()
# classify(model,x_train,x_test,y_train,y_test)
# print('----------- Extra Trees --------------')
# model = ExtraTreesClassifier()
# classify(model,x_train,x_test,y_train,y_test)
# print('----------- AdaBoost --------------')
# model = AdaBoostClassifier()
# classify(model,x_train,x_test,y_train,y_test)
# print('----------- Gradient Boost --------------')
# GradientBoostingClassifier()
# classify(model,x_train,x_test,y_train,y_test)
# print('----------- KNN --------------')
# model = KNeighborsClassifier()
# classify(model,x_train,x_test,y_train,y_test)
# print('----------- SGD --------------')
# model = SGDClassifier()
# classify(model,x_train,x_test,y_train,y_test)
# print('----------- Logisitc Regression --------------')
# model = LogisticRegression()
# classify(model,x_train,x_test,y_train,y_test)
# print('----------- Decision Tree --------------')
# model = DecisionTreeClassifier()
# classify(model,x_train,x_test,y_train,y_test)

#################################################
#             Hyperparameter Tuning             #
#################################################
param_grid = {
    'n_estimators':[50,100,150,200],  # most times was 100
    'criterion':['gini','entropy'],
    'max_depth':[10,15,16,17,18],
    'min_samples_split':[10,12,13,20],   # first 3 times was 12
    # 'min_samples_leaf':[3,4,5],
    'max_features':['sqrt','log2'],
    # 'max_leaf_nodes':[1,5,10,15],
    # 'min_impurity_decrease':[0,0.1,0.3],
    # 'min_weight_fraction_leaf':[0,0.1,0.3],
    # 'min_impurity_split':[0,0.1,0.3],
    # 'bootstrap':[True,False], # first 2 times was False
    # 'verbose':[1,5,10],
    # 'warm_start':[True,False],
    # 'class_weight':[None,'balanced'],
    'ccp_alpha':[0,0.001,0.003,0.1,0.3,0.5,1],
    'max_samples':[0.5,0.7,0.8,1,5,10]
}

# def accuracy(confusion):
#     TP = confusion[0,0] # True Positive
#     FP = confusion[0,1] # False Positive
#     FN = confusion[1,0] # False Negative
#     TN = confusion[1,1] # True Negative
#     return (100*(TP+TN)/(TP+FP+FN+TN))

# def score(y_pred,y):
#     confusion = confusion_matrix(y,y_pred)
#     score = accuracy(confusion)
#     return score

# scorer = make_scorer(score)
# scores = []
# model = RandomForestClassifier()
# clf = GridSearchCV(estimator=model,param_grid=param_grid,scoring='accuracy',cv=5,verbose=10,n_jobs=-1)
# # clf = GridSearchCV(estimator=model,param_grid=param_grid,cv=5,scoring=scorer,verbose=10,n_jobs=-1)
# clf.fit(x_train,y_train)
# scores.append({
#     'best_score':clf.best_score_,
#     'best_params':clf.best_params_,
#     'best_grid':clf.best_estimator_
# })
# print(scores)

# pd.set_option('display.max_colwidth',1)
# res = pd.DataFrame(scores,columns=['model','best_score','best_params'])
# print(res)

#################################################
#    Real-Time videostream and classification   #
#################################################
probabilities = []
secs = []
score = 0
i = 0

while True:
    ret, frame = vs.read()   # read video frame by frame

    frame = resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    rects = detector(gray, 0)   # detect a face
    for rect in rects:  # run on all faces found
        i += 1
        if i % 20 == 0: # analyze frame approximately every second
            shape = predictor(gray, rect)   # get 68 points representing the face
            shape = face_utils.shape_to_np(shape)   # convert to array
            features = []
            nose = shape[32:68]
            eye = shape[36:68]
            ear = eye_aspect_ratio(eye)
            mar = mouth_aspect_ratio(eye)
            cir = circularity(eye)
            mouth_eye = mouth_over_eye(eye)
            head = head_drop(nose)
            features.append([ear, mar, cir, mouth_eye,head])
            # print(features)
            predictions = model.predict_proba(features) # make prediction on frame
            print(predictions[0][1])
            probabilities.append(predictions[0][1]) # probability to be drowsy 
            if len(probabilities) == 6: # sum probabilities of last 5 frames
                probabilities.pop(0)
            seq = sum(probabilities)
            if seq > 4:
                score = 1
            elif seq > 3:
                score += 0.5
            elif seq > 2:
                score += 0.25
            else:
                score -= 0.1
            if score > 1:
                score = 1
            elif score < 0:
                score = 0
            if score > 0.75: print('Sleeping')
            elif score > 0.4:
                print('Drowsy')
            else:
                print('Alert')

    cv2.imshow("Frame", frame)  # show frame
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"): 
        break

cv2.destroyAllWindows()
vs.stop()