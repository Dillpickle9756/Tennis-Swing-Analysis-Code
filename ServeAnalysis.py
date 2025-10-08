from sympy.codegen.scipy_nodes import cosm1
from ultralytics import YOLO
import cv2
import numpy as np
from scipy.interpolate import CubicSpline
import argparse
import matplotlib.pyplot as plt


#Insert your path and file name here ⬇
# ----------------------------
# Parse Command-Line Arguments
# ----------------------------
parser = argparse.ArgumentParser(description="Analyze tennis swing from video using YOLO models.")
parser.add_argument("--FileName", type=str, required=True,
                    help="Path to the input video file (.MOV, .MP4, etc.)")
parser.add_argument("--Hand", type=str, choices=["Left", "Right"], required=True,
                    help="Dominant hand of the player (Left or Right)")
parser.add_argument("--PoseModel", type=str, required=True,
                    help="Path to YOLO pose model (e.g., yolo11s-pose.pt)")
parser.add_argument("--DetectionModel", type=str, required=True,
                    help="Path to YOLO detection model (e.g., best.pt)")

args = parser.parse_args()

# ----------------------------
# Assign Variables
# ----------------------------
FileName = args.FileName
Hand = args.Hand
PoseModelPath = args.PoseModel
DetectionModelPath = args.DetectionModel

# ----------------------------
# Initialize Video + Models
# ----------------------------
vid = cv2.VideoCapture(FileName)
poseModel = YOLO(PoseModelPath)
DetectionModel = YOLO(DetectionModelPath)
isRightMovingAvg = None
epsilon = 0.9
LowestRightLegAngle = 300
LowestLeftLegAngle = 300
RightArmAngles = []
LeftArmAngles = []
RacketHeights = []
RightLegAngles = []
LeftLegAngles = []
SwingScore = 0
BodyScore = 0
OverallScore = 0

RightFootHeight = 0
HighestRightFootHeight = 3000
RightFeet = []
HighestLeftFootHeight = 3000
LeftFootHeight = 0
LeftFeet = []
IndexList = []
count = 0
while True:
    ret, frame = vid.read()
    if ret:
        if FileName[56] == "I":
            frameTransposed = np.fliplr(np.transpose(frame,(1,0,2))).astype(np.uint8).copy()
        else:
            frameTransposed = frame
# Predict with the model
        poseResults = poseModel(frameTransposed)  # predict on an image
        detectionResults = DetectionModel.predict(frameTransposed, conf=0.5)
# Access the results
        racketBox = None
        ballBox = None
        bestRacketConf = 0
        bestBallConf = 0
        for result in detectionResults:
            boxes = result.boxes  # Boxes object for bounding box outputs
            for box in boxes:
                if box.cls.item() == 1:
                    if box.conf.item() > bestRacketConf:
                        bestRacketConf = box.conf.item()
                        racketBox = box.xyxy.numpy()[0]
                if box.cls.item() == 0:
                    if box.conf.item() > bestBallConf:
                        bestBallConf = box.conf.item()
                        ballBox = box.xyxy.numpy()[0]
        if racketBox is not None:
            cv2.rectangle(frameTransposed, (int(round(racketBox[0])), int(round(racketBox[1]))),(int(round(racketBox[2])), int(round(racketBox[3]))), (0, 255, 0), 4)
        if ballBox is not None:
            cv2.rectangle(frameTransposed, (int(round(ballBox[0])), int(round(ballBox[1]))),(int(round(ballBox[2])), int(round(ballBox[3]))), (0, 255, 0), 4)

        for result in poseResults:
            xy = result.keypoints.xy[0].numpy()  # x and y coordinates
            #xyn = result.keypoints.xyn  # normalized
            kpts = result.keypoints.data[0].numpy()  # x, y, visibility (if available)
            Visibility = 0.7
            rightLeg = kpts[[12,14,16]]
            rightHip = rightLeg[0]
            rightKnee = rightLeg[1]
            rightFoot = rightLeg[2]
            leftLeg = kpts[[11,13,15]]
            leftHip = leftLeg[0]
            leftKnee = leftLeg[1]
            leftFoot = leftLeg[2]
            rightArm = kpts[[6, 8, 10]]
            rightShoulder = rightArm[0]
            rightElbow = rightArm[1]
            rightWrist = rightArm[2]
            leftArm = kpts[[5, 7, 9]]
            leftShoulder = leftArm[0]
            leftElbow = leftArm[1]
            leftWrist = leftArm[2]
        if Hand == "Right":
            if rightHip[2] > Visibility and rightKnee[2] > Visibility and rightFoot[2] > Visibility:
                rightlowerLeg = rightFoot[:2] - rightKnee[:2]
                rightupperLeg = rightHip[:2] - rightKnee[:2]
                rightlowerLeg/=np.linalg.norm(rightlowerLeg)
                rightupperLeg/=np.linalg.norm(rightupperLeg)
                cos = rightlowerLeg[0] * rightupperLeg[0] + rightlowerLeg[1] * rightupperLeg[1]
                rightkneeAngle = np.arccos(cos) * 180/np.pi
                RightLegAngles.append(rightkneeAngle)
                if rightkneeAngle < LowestRightLegAngle:
                    LowestRightLegAngle = rightkneeAngle
                RightFootHeight = rightFoot[1]
                RightFeet.append(round(RightFootHeight))
            if rightShoulder[2] > Visibility and rightElbow[2] > Visibility and rightWrist[2] > Visibility and racketBox is not None:
                Forearm = rightWrist[:2] - rightElbow[:2]
                upperArm = rightShoulder[:2] - rightElbow[:2]
                Forearm /= np.linalg.norm(Forearm)
                upperArm /= np.linalg.norm(upperArm)
                cos = Forearm[0] * upperArm[0] + Forearm[1] * upperArm[1]
                RightelbowAngle = np.arccos(cos) * 180 / np.pi
                RightArmAngles.append(round(RightelbowAngle))
                racketHeight = racketBox[1]
                RacketHeights.append(round(racketHeight))
                IndexList.append(count)
            if leftFoot[2] > Visibility:
                LeftFootHeight = leftFoot[1]
                LeftFeet.append(LeftFootHeight)
                if LeftFootHeight < HighestLeftFootHeight:
                    HighestLeftFootHeight = LeftFootHeight
        if Hand == "Left":
            if leftHip[2] > Visibility and leftKnee[2] > Visibility and leftFoot[2] > Visibility:
                leftlowerLeg = leftFoot[:2] - leftKnee[:2]
                leftupperLeg = leftHip[:2] - leftKnee[:2]
                leftlowerLeg/=np.linalg.norm(leftlowerLeg)
                leftupperLeg/=np.linalg.norm(leftupperLeg)
                cos = leftlowerLeg[0] * leftupperLeg[0] + leftlowerLeg[1] * leftupperLeg[1]
                leftkneeAngle = np.arccos(cos) * 180/np.pi
                LeftLegAngles.append(leftkneeAngle)
                if leftkneeAngle < LowestLeftLegAngle:
                    LowestLeftLegAngle = leftkneeAngle
                LeftFootHeight = leftFoot[1]
                LeftFeet.append(round(LeftFootHeight))
            if leftShoulder[2] > Visibility and leftElbow[2] > Visibility and leftWrist[2] > Visibility and racketBox is not None:
                Forearm = leftWrist[:2] - leftElbow[:2]
                upperArm = leftShoulder[:2] - leftElbow[:2]
                Forearm /= np.linalg.norm(Forearm)
                upperArm /= np.linalg.norm(upperArm)
                cos = Forearm[0] * upperArm[0] + Forearm[1] * upperArm[1]
                LeftelbowAngle = np.arccos(cos) * 180 / np.pi
                LeftArmAngles.append(round(LeftelbowAngle))
                racketHeight = racketBox[1]
                RacketHeights.append(round(racketHeight))
                IndexList.append(count)
            # Metric: Height of Ball Toss
            # Metric: Angle of arm when Racket reaches max height
            if leftFoot[2] > Visibility:
                RightFootHeight = rightFoot[1]
                RightFeet.append(RightFootHeight)
                if RightFootHeight < HighestRightFootHeight:
                    HighestRightFootHeight = RightFootHeight

        for x,y,k in kpts:
            if k > Visibility:
                cv2.circle(frameTransposed,(int(round(x)),int(round(y))),4,(0,0,0),-1)
        cv2.imshow('Video Frame', frameTransposed)
        cv2.waitKey(1000//30)
    count = count + 1
    if not ret:
        print("break")
        break
if Hand == "Right":
    i = 0
    index = 0
    HighestRightFootHeight = RightFeet[0]
    while i < len(RightFeet):
        if RightFeet[i] < HighestRightFootHeight:
            HighestRightFootHeight = RightFeet[i]
            index = i
        i = i+1
    print("Leg angle at followthrough: " + str(RightLegAngles[index]))


    Cubic_Spline = CubicSpline(np.asarray(IndexList,dtype = np.float64),np.asarray(RacketHeights, dtype = np.float64))
    xs = np.arange(count, dtype = np.float64)
    InterpolatedRacketHeights = Cubic_Spline(xs)
    #plt.scatter(IndexList,RacketHeights)
    #plt.plot(xs,InterpolatedRacketHeights)
    #plt.show()
    Cubic_Spline = CubicSpline(np.asarray(IndexList,dtype = np.float64),np.asarray(RightArmAngles, dtype = np.float64))
    xs = np.arange(count, dtype = np.float64)
    InterpolatedRightArmAngles = Cubic_Spline(xs)





    HighestRacket = InterpolatedRacketHeights[0]
    i = 0
    ind = 0
    while i < len(InterpolatedRacketHeights):
        if InterpolatedRacketHeights[i] < HighestRacket:
            HighestRacket = InterpolatedRacketHeights[i]
            ind = i
        i = i + 1
    print("Arm Angle At Contact: " + str(InterpolatedRightArmAngles[ind]))
    SwingScore = (round(InterpolatedRightArmAngles[ind]) + 20)
    SwingScore = SwingScore - 100
    BodyScore1 = (round(LowestRightLegAngle) + 20)
    BodyScore1 = BodyScore1 - 90
    BodyScore1 = abs(BodyScore1 - 100)
    BodyScore2 = abs((round(RightLegAngles[index]) + 20) - 90)
    BodyScore = (BodyScore1 + BodyScore2)/2
    print("Swing Score: " + str(SwingScore))
    print("Body Score: " + str(BodyScore))
    OverallScore = round((BodyScore + SwingScore) / 2)
    print("Your overall serve score was: " + str(OverallScore))
    if RightArmAngles[ind] < 160:
        print("You hit the ball too low. Try swinging earlier to catch the ball before it starts falling.")
    if RightArmAngles[ind] >= 160:
        print("The height of your ball toss was good")
    if LowestRightLegAngle > 110:
        print("You can load your legs more to get more power on your serve")
    if LowestRightLegAngle < 110:
        print("Your leg load is good")
if Hand == "Left":
    i = 0
    index = 0
    HighestLeftFootHeight = LeftFeet[0]
    while i < len(LeftFeet):
        if LeftFeet[i] < HighestLeftFootHeight:
            HighestLeftFootHeight = LeftFeet[i]
            index = i
        i = i+1
    print("Leg angle at followthrough: " + str(LeftLegAngles[index]))

    Cubic_Spline = CubicSpline(np.asarray(IndexList, dtype=np.float64), np.asarray(RacketHeights, dtype=np.float64))
    xs = np.arange(count, dtype=np.float64)
    InterpolatedRacketHeights = Cubic_Spline(xs)

    Cubic_Spline = CubicSpline(np.asarray(IndexList,dtype = np.float64),np.asarray(LeftArmAngles, dtype = np.float64))
    xs = np.arange(count, dtype = np.float64)
    InterpolatedLeftArmAngles = Cubic_Spline(xs)


    HighestRacket = RacketHeights[0]
    i = 0
    ind = 0
    while i < len(RacketHeights):
        if RacketHeights[i] < HighestRacket:
            HighestRacket = RacketHeights[i]
            ind = i
        i = i+1
    print("Arm Angle At Contact: " + str(LeftArmAngles[ind]))
    SwingScore = (round(LeftArmAngles[ind]) + 20)
    SwingScore = SwingScore - 100
    BodyScore1 = (round(LowestLeftLegAngle) + 20)
    BodyScore1 = BodyScore1 - 90
    BodyScore1 = abs(BodyScore1 - 100)
    BodyScore2 = abs((round(LeftLegAngles[index]) + 20) - 90)
    BodyScore = (BodyScore1 + BodyScore2) / 2
    print("Swing Score: " + str(SwingScore))
    print("Body Score: " + str(BodyScore))
    OverallScore = round((BodyScore + SwingScore) / 2)
    print("Your overall serve score was: " + str(OverallScore))
    if LeftArmAngles[ind] < 160:
        print("You hit the ball too low. Try swinging earlier to catch the ball before it starts falling.")
    if LeftArmAngles[ind] >= 160:
        print("The height of your ball toss was good")
    if LowestLeftLegAngle > 110:
        print("You can load your legs more to get more power on your serve")
    if LowestLeftLegAngle < 110:
        print("Your leg load is good")