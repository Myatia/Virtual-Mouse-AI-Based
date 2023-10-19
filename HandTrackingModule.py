import cv2
import mediapipe as mp
import time

capture = cv2.VideoCapture(0)

mpHands = mp.solutions.hands   #formalities before using media pipe
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

#for framerate showing setting 0 as initial
previousTime = 0
currentTime = 0

while True:
    success, img = capture.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    # open the object
    if results.multi_hand_landmarks:
        for handLandMarks in results.multi_hand_landmarks:  #handLandMarks in this line is single hand
            for id, lm in enumerate(handLandMarks.landmark):
                #print(id, lm)
                height, width, channel = img.shape
                cx, cy = int(lm.x*width), int(lm.y*height)     #finding position
                print(id, cx, cy)

                if id == 0:
                    cv2.circle(img, (cx, cy), 10, (255,255,255), cv2.FILLED)


            mpDraw.draw_landmarks(img, handLandMarks, mpHands.HAND_CONNECTIONS)  #HAND_CONNECTIONS - for connecting dots on hands

    #getting fps for latency measurement
    currentTime = time.time()  #getting current time
    fps = 1/(currentTime-previousTime)
    previousTime = currentTime

    #displaying fps on screen
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,0), 3)


    cv2.imshow("WebCam Capture", img)
    cv2.waitKey(1)