from sympy.codegen.scipy_nodes import cosm1
from ultralytics import YOLO
import cv2
import numpy as np


#Insert your path and file name here ⬇

FileName = '/Users/dhilanbelur/Desktop/Tennis App/Back View/Labeled/IMG_2832.mov'
vid = cv2.VideoCapture(FileName)
Rackets = []

#Fill this in with either "Left" or "Right"

Hand ="Right"


LowestRightLegAngle = 300
LowestLeftLegAngle = 300
# Load a model
poseModel = YOLO('/Users/dhilanbelur/Desktop/Tennis App/yolo11s-pose.pt')  # load an official model
DetectionModel = YOLO("/Users/dhilanbelur/PycharmProjects/TennisCoachingApp/runs/detect/train2/weights/best.pt")  # build from YAML and transfer weights

while True:
    ret, frame = vid.read()
    if ret:
        if FileName[56] == "I":
            frameTransposed = np.fliplr(np.transpose(frame,(1,0,2))).astype(np.uint8).copy()
        else:
            frameTransposed = frame
# Predict with the model
        poseResults = poseModel(frameTransposed)  # predict on an image
        detectionResults = DetectionModel.predict(frameTransposed, conf=0.1)
# Access the results
        for result in poseResults:
            xy = result.keypoints.xy[0].numpy()  # x and y coordinates
            #xyn = result.keypoints.xyn  # normalized
            kpts = result.keypoints.data[0].numpy()  # x, y, visibility (if available)
            Visibility = 0.7
            rightArm = kpts[[6,8,10]]
            rightShoulder = rightArm[0]
            rightElbow = rightArm[1]
            rightWrist = rightArm[2]
            leftArm = kpts[[5, 7, 9]]
            leftShoulder = leftArm[0]
            leftElbow = leftArm[1]
            leftWrist = leftArm[2]
            rightLeg = kpts[[12, 14, 16]]
            rightHip = rightLeg[0]
            rightKnee = rightLeg[1]
            rightFoot = rightLeg[2]
            leftLeg = kpts[[11, 13, 15]]
            leftHip = leftLeg[0]
            leftKnee = leftLeg[1]
            leftFoot = leftLeg[2]
            for x,y,k in kpts:
                if k > Visibility:
                    cv2.circle(frameTransposed,(int(round(x)),int(round(y))),6,(0,0,0),-1)
        if Hand == "Right":
            if rightShoulder[2] > Visibility and rightElbow[2] > Visibility and rightWrist[2] > Visibility:
                #Arm Angle Calculations
                Forearm = rightWrist[:2] - rightElbow[:2]
                upperArm = rightShoulder[:2] - rightElbow[:2]
                Forearm/=np.linalg.norm(Forearm)
                upperArm/=np.linalg.norm(upperArm)
                cos = Forearm[0] * upperArm[0] + Forearm[1] * upperArm[1]
                elbowAngle = np.arccos(cos) * 180/np.pi
            if rightHip[2] > Visibility and rightKnee[2] > Visibility and rightFoot[2] > Visibility:
                rightlowerLeg = rightFoot[:2] - rightKnee[:2]
                rightupperLeg = rightHip[:2] - rightKnee[:2]
                rightlowerLeg /= np.linalg.norm(rightlowerLeg)
                rightupperLeg /= np.linalg.norm(rightupperLeg)
                cos = rightlowerLeg[0] * rightupperLeg[0] + rightlowerLeg[1] * rightupperLeg[1]
                rightkneeAngle = np.arccos(cos) * 180 / np.pi
                if rightkneeAngle < LowestRightLegAngle:
                    LowestRightLegAngle = rightkneeAngle
        if Hand == "Left":
            if leftShoulder[2] > Visibility and leftElbow[2] > Visibility and leftWrist[2] > Visibility:
                Forearm = leftWrist[:2] - leftElbow[:2]
                upperArm = leftShoulder[:2] - leftElbow[:2]
                Forearm /= np.linalg.norm(Forearm)
                upperArm /= np.linalg.norm(upperArm)
                cos = Forearm[0] * upperArm[0] + Forearm[1] * upperArm[1]
                elbowAngle = np.arccos(cos) * 180 / np.pi
            if leftHip[2] > Visibility and leftKnee[2] > Visibility and leftFoot[2] > Visibility:
                leftlowerLeg = leftFoot[:2] - leftKnee[:2]
                leftupperLeg = leftHip[:2] - leftKnee[:2]
                leftlowerLeg/=np.linalg.norm(leftlowerLeg)
                leftupperLeg/=np.linalg.norm(leftupperLeg)
                cos = leftlowerLeg[0] * leftupperLeg[0] + leftlowerLeg[1] * leftupperLeg[1]
                leftkneeAngle = np.arccos(cos) * 180/np.pi
                if leftkneeAngle < LowestLeftLegAngle:
                    LowestLeftLegAngle = leftkneeAngle
        racketBox = None
        bestRacketConf = 0
        bestBallConf = 0
        for result in detectionResults:
            boxes = result.boxes  # Boxes object for bounding box outputs
            for box in boxes:
                if box.cls.item() == 1:
                    if box.conf.item() > bestRacketConf:
                        bestRacketConf = box.conf.item()
                        racketBox = box.xyxy.numpy()[0]
            for box in boxes:
                if box.cls.item() == 0:
                    if box.conf.item() > bestBallConf:
                        bestBallConf = box.conf.item()
                        ballBox = box.xyxy.numpy()[0]
            if racketBox is not None:
                Rackets.append(racketBox)
                cv2.rectangle(frameTransposed, (int(round(racketBox[0])),int(round(racketBox[1]))),(int(round(racketBox[2])),int(round(racketBox[3]))),(0,255,0),4)
        i = 0
        while i <= (len(Rackets) - 1):
            X_Center = ((int(round(Rackets[i][2] + Rackets[i][0])/2)))
            Y_Center = ((int(round(Rackets[i][3] + Rackets[i][1])/2)))
            cv2.circle(frameTransposed, (X_Center, Y_Center), 4, (0,0,255), -1)
            i = i + 1
        cv2.imshow('Video Frame', frameTransposed)
        cv2.waitKey(1000//30)
    if not ret:
        break

if Hand == "Left":
    if LowestLeftLegAngle < 120:
        print("Your leg load is good on your forehand.")
    else:
        print("You could load your legs more to get more power on your forehand")
if Hand == "Right":
    if LowestRightLegAngle < 120:
        print("Your leg load is good on your forehand.")
    else:
        print("You could load your legs more to get more power on your forehand")