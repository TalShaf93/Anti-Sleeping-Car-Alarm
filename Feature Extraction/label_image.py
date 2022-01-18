##########################################################
#                     Imported Libraries                 #
##########################################################
import cv2
import os
from csv import writer
import csv

##########################################################
#               Find new video to label                  #
##########################################################
directory = 'D:/Drv-Dev/Python/13- Final Project/Videos'
data = []
for filename in os.listdir(directory):  # run on all files in folder
    checked = False             # boolean variable expressing if video has been checked before
    x0 = filename.split('.')
    video_num = x0[0]   # video number

    # check if video has been seen before
    with open('training.csv') as file:  
        reader = csv.reader(file)
        for row in reader:
            if video_num in row[0]: 
                checked = True
                break

    # prepare video for labeling           
    if not checked: 
        array = []
        video = cv2.VideoCapture((os.path.join(directory, filename)))   # read video
        success = True
        count = 0
        fps = video.get(cv2.CAP_PROP_FPS)   # frames per second
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # number of frames in video
        print(fps,frame_count)

        # label video
        while count < frame_count and success:
            success, frame = video.read()   # read video frame by frame
            
            if count % (int(fps)/2) == 0:    # take 2 frames per second
                # print('saved image')
                cv2.imwrite('Frames/{filename}-{index}.png'.format(filename=filename,index=count),frame)    # save frame
                img = cv2.imread('Frames/{filename}-{index}.png'.format(filename=filename,index=count)) # read image
                height, width, channels = img.shape # size of image
                img = cv2.resize(img,(int(width*0.7),int(height*0.7)))    # resize image
                cv2.imshow('img',img)   # display image
                cv2.waitKey(1)
                label = input('Drowsy=1, Alert=0: ')
                cv2.destroyAllWindows()
                data.append([video_num,count,label])
                print(count/(int(fps)/2))
            count += 1
    else:
        print('already checked video {num}'.format(num=video_num))

##########################################################
#                   add all data to csv                  #
##########################################################
with open('training.csv','a',newline='') as f:
    thewriter = csv.writer(f)
    for example in data:
        thewriter.writerow(example)



