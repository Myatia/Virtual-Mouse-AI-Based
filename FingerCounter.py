import cv2
import time
import os
import HandTrackingModule as htm

widthCam, heightCam = 640, 480

capture = cv2.VideoCapture(0)
capture.set(3, widthCam)
capture.set(4, heightCam)

folderPath = "images"  # folderPath for images folder
myList = os.listdir(folderPath)
print(myList)
overlayList = []  # list of images that will overlay upon the webcam capture video
for imagePath in myList:  # imagePath for each image
    image = cv2.imread(f'{folderPath}/{imagePath}')
    overlayList.append(image)

pTime = 0

detector = htm.handDetector(minDetectionConfidence=0.75)

tipIDs = [4, 8, 12, 16, 20]

while True:
    success, img = capture.read()
    img = detector.findHands(img)
    lmlist = detector.findPosition(img, draw=False)

    if len(lmlist) != 0:
        fingers = []

        # thumb condition for left hand
        # change greater than for right hand
        if lmlist[tipIDs[0]][1] < lmlist[tipIDs[0] - 1][1]:  # 2 is y-axis
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):  # loop for four fingers, thumb not included
            # if lmlist[8][2] < lmlist[6][2]:
            if lmlist[tipIDs[id]][2] < lmlist[tipIDs[id] - 2][2]:  # 2 is y-axis
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)

        h, w, c = overlayList[totalFingers - 1].shape
        img[0:h, 0:w] = overlayList[totalFingers - 1]  # first 0:200 is height, later is width

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Webcam", img)
    cv2.waitKey(1)
