import cv2
import time

import math

# wCam, hCam = 1500, 2000
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)
# cap.set(3, wCam)
# cap.set(4, hCam)
classNames = []
pTime = 0
dcount = 0
frame = 0
tcount = 0

classFile = '/Users/iklak/PycharmProjects/pythonProject2/yolo/darknet/data/coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
configpath = '/Users/iklak/PycharmProjects/pythonProject2/yolov3-tiny.cfg'
weightspath = '/Users/iklak/PycharmProjects/pythonProject2/yolov3-tiny.weights'

net = cv2.dnn_DetectionModel(weightspath, configpath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# out = cv2.VideoWriter('test.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)
while True:
    ret, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=0.2)

    for classId, confidence, box in zip(classIds, confs, bbox):
        if (confidence * 100) > 10:
            print(box)

            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            # cv2.putText(img, classNames[classId-1],(box[0]+10),(box[1]+30),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),2)

            # print(classIds, bbox, classNames[classId])
    cv2.imshow("test",img)
    cv2.waitKey(1)