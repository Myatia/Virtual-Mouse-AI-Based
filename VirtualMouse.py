import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

# 1. finding hand landmarks
# 2. getting the tip of the index and middle fingers
# 3. checking which fingers are up
# 4. moving mode checking: index finger up
    # 4.1. converting coordinates
# 5. smoothing the values
# 6. moving the mouse
# 7. clicking mode checking: index and middle fingers up
# 8. finding distance between fingers
# 9. clicking when distance is short
# 10. showing frames rate
# 11. displaying

# variable declaration

widthCam, heightCam = 640, 480
previousTime = 0
widthScreen, heightScreen = autopy.screen.size()
frameReduction = 100
smoothening = 5
previousLocationX, previousLocationY = 0, 0
currentLocationX, currentLocationY = 0, 0
scale = autopy.screen.scale()

# end of variable declaration

capture = cv2.VideoCapture(0)
capture.set(3, widthCam)  # width setting
capture.set(4, heightCam)  # height setting
detector = htm.handDetector(maxNumberHands=1)


while True:
    # 1. finding hand landmarks
    success, img = capture.read()
    flip = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmlist = detector.findPosition(img)
    # print(lmlist)

    # 2. getting the tip of the index and middle fingers
    if len(lmlist) != 0:
        x1, y1 = lmlist[8][1:]    # index
        x2, y2 = lmlist[12][1:]   # middle
        #print(x1,y1, x2,y2)


        # 3. checking which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)
        cv2.rectangle(img, (frameReduction, frameReduction), (widthCam - frameReduction, heightCam - frameReduction), (255, 0, 255), 2)

        # 4. moving mode checking: index finger up
        if fingers[1] == 1 and fingers[2] == 0:
            # 4.1. converting coordinates
            x3 = np.interp(x1, (frameReduction, widthCam - frameReduction), (0, widthScreen))
            y3 = np.interp(y1, (frameReduction, heightCam - frameReduction), (0, heightScreen))

            # 5. smoothing the values
            currentLocationX = previousLocationX + (x3 - previousLocationX) / smoothening
            currentLocationY = previousLocationY + (y3 - previousLocationY) / smoothening

            # 6. moving the mouse
            autopy.mouse.move((widthScreen - currentLocationX) / scale, (currentLocationY) / scale)
            #autopy.mouse.move(widthScreen - currentLocationX, currentLocationY)
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            previousLocationX, previousLocationY = currentLocationX, currentLocationY

        # 7. clicking mode checking: index and middle fingers up
        if fingers[1] == 1 and fingers[2] == 1:
            # 8. finding distance between fingers
            length, img, infoLine = detector.findDistance(8, 12, img)
            # print(length)
            # 9. clicking when distance is short
            if length < 30:
                cv2.circle(img, (infoLine[4], infoLine[5]), 10, (0, 255, 255), cv2.FILLED)
                autopy.mouse.click()


    # 10. showing frames rate
    currentTime = time.time()
    fps = 1/(currentTime-previousTime)
    previousTime = currentTime
    cv2.putText(img, str(int(fps)), (20,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0, 0), 3)

    # 11. displaying
    cv2.imshow("Webcam", img)
    cv2.waitKey(1)
