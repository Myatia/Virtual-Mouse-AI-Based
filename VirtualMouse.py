import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy
import pyautogui
from pynput.mouse import Listener
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# 1. finding hand landmarks
# 2. getting the tip of the fingers
# 3. checking which fingers are up
# 4. moving mode checking: index finger up
    # 4.1. converting coordinates
    # 4.2 smoothing the values
    # 4.3 moving the mouse
# 5. clicking mode checking: index and middle fingers up
    # 5.1 finding distance between fingers
    # 5.2 clicking when distance is short
#
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
totalClick = 0
falseClick = 0
# totalClickCount = 0
# actualClickCount = 0

# end of variable declaration

capture = cv2.VideoCapture(0)
capture.set(3, widthCam)  # width setting
capture.set(4, heightCam)  # height setting
detector = htm.handDetector(maxNumberHands=1)

# volume control initialization library

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()  # -50 to 5

minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0

# end of volume control

while True:
    # 1. finding hand landmarks
    success, img = capture.read()
    # flip = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmlist = detector.findPosition(img)
    # print(lmlist)

    # def countMouseClick():
    #     global totalClickCount
    #     totalClickCount += 1
    #     print(f'Mouse total click count: {totalClickCount}')
    #
    # def ClickCount(x, y, button, pressed):
    #     global actualClickCount
    #     if pressed:
    #         if button == button.left:
    #             actualClickCount += 1
    #             print(f'Actual left click count: {actualClickCount}')


    # with Listener(on_click = actualClickCount) as listener:
    #     listener.join()

    # 2. getting the tip of the fingers
    if len(lmlist) != 0:
        x1, y1 = lmlist[8][1:]    # index
        x2, y2 = lmlist[12][1:]   # middle
        x4, y4 = lmlist[4][1:]    # thumb
        x5, y5 = lmlist[16][1:]   # ring
        x6, y6 = lmlist[20][1:]   # pinky
        #print(x1,y1, x2,y2)

        # Ground truth landmarks
        # ground_truth = np.array([[x1, y1], [x2, y2], [x4, y4], [x5, y5], [x6, y6]])
        # print(ground_truth)

        # 3. checking which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)
        cv2.rectangle(img, (frameReduction, frameReduction), (widthCam - frameReduction, heightCam - frameReduction), (255, 0, 255), 2)

        # 4. moving mode checking: index finger up
        if fingers[1] == 1 and fingers[2] == 0 and fingers[0] == 0:
            # 4.1. converting coordinates
            x3 = np.interp(x1, (frameReduction, widthCam - frameReduction), (0, widthScreen))
            y3 = np.interp(y1, (frameReduction, heightCam - frameReduction), (0, heightScreen))

            # 4.2. smoothing the values
            currentLocationX = previousLocationX + (x3 - previousLocationX) / smoothening
            currentLocationY = previousLocationY + (y3 - previousLocationY) / smoothening

            # 4.3. moving the mouse
            autopy.mouse.move((widthScreen - currentLocationX) / scale, (currentLocationY) / scale)
            # autopy.mouse.move(widthScreen - currentLocationX, currentLocationY)
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            previousLocationX, previousLocationY = currentLocationX, currentLocationY

        # 5. clicking mode checking (left click): index and middle fingers up
        if fingers[1] == 1 and fingers[2] == 1:
            # 5.1 finding distance between fingers
            length, img, infoLine = detector.findDistance(8, 12, img)
            # print(length)
            # 5.2 clicking when distance is short
            if length < 30:
                cv2.circle(img, (infoLine[4], infoLine[5]), 10, (0, 255, 255), cv2.FILLED)
                # 5.3 exception handling and for calculating accuracy
                try:
                    autopy.mouse.click()
                    totalClick += 1
                except autopy.autopy.error.MouseError:
                    falseClick += 1
                    print("False left click happen")

                # click events counter
                # print(f'Total clicks: {totalClick}')
                # print(f'False clicks: {falseClick}')

                # countMouseClick()

        # 6. clicking mode checking (right click): index and middle fingers up
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
            # 6.1 finding distance between fingers
            length, img, infoLine = detector.findDistance(8, 12, img)
            length1, img, infoLine1 = detector.findDistance(12, 16, img)

            # 6.2 clicking when distance is short
            if length < 30 and length1 < 30:
                cv2.circle(img, (infoLine[4], infoLine[5]), 10, (0, 255, 255), cv2.FILLED)
                autopy.mouse.click(autopy.mouse.Button.RIGHT)

        # 7. scrolling up
        if fingers[0] == 1 and fingers[1] == 0:
            cv2.circle(img, (x4, y4), 10, (255, 0, 255), cv2.FILLED)
            pyautogui.scroll(50)  # scroll up 50 "clicks"

        # 8. scrolling down
        if fingers[4] == 1:
            cv2.circle(img, (x6, y6), 10, (255, 0, 255), cv2.FILLED)
            pyautogui.scroll(-50)  # scroll down 50 "clicks"

        # 9. volume up and down
        if fingers[0] == 1 and fingers[1] == 1:
            length, img, infoLine = detector.findDistance(4, 8, img)

            # Hand range 50 to 250
            # Volume range -50 to 5
            vol = np.interp(length, [50, 250], [minVol, maxVol])
            volBar = np.interp(length, [50, 250], [400, 150])
            volPer = np.interp(length, [50, 250], [0, 100])
            # print(int(length), vol)
            volume.SetMasterVolumeLevel(vol, None)

            if length < 50:
                cv2.circle(img, (infoLine[4], infoLine[5]), 15, (0, 255, 0), cv2.FILLED)

            # Drawings for volume bar
            cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
            cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
            cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                        1, (255, 0, 0), 3)

    # 10. showing frames rate
    currentTime = time.time()
    fps = 1/(currentTime-previousTime)
    previousTime = currentTime
    cv2.putText(img, str(int(fps)), (20,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0, 0), 3)

    # 11. displaying
    cv2.imshow("Webcam", img)
    cv2.waitKey(1)
