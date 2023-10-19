import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm


# for frame rate showing setting 0 as initial
previous_time = 0
current_time = 0

capture = cv2.VideoCapture(0)

detector = htm.handDetector()

while True:
    success, img = capture.read()
    img = detector.findHands(img)
    lmlist = detector.findPosition(img)
    if len(lmlist) != 0:
        print(lmlist[4])

    # getting fps for latency measurement
    current_time = time.time()  # getting current time
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    # displaying fps on screen
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)

    cv2.imshow("WebCam Capture", img)
    cv2.waitKey(1)