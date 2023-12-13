import cv2
import mediapipe as mp
import time
import math


class handDetector():
    def __init__(self, mode=False, maxNumberHands=2, modelComplexity=1, minDetectionConfidence=0.5,
                 minTrackingConfidence=0.5):
        # self.lmList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        self.mode = mode
        self.maxNumberHands = maxNumberHands
        self.modelComplexity = modelComplexity
        self.minDetectionConfidence = minDetectionConfidence
        self.minTrackingConfidence = minTrackingConfidence

        self.mpHands = mp.solutions.hands  # formalities before using media pipe
        self.hands = self.mpHands.Hands(self.mode, self.maxNumberHands, self.modelComplexity,
                                        self.minDetectionConfidence, self.minTrackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIDs = [4, 8, 12, 16, 20]
        self.leftHand = True

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        # open the object
        if self.results.multi_hand_landmarks:
            for handLandMarks in self.results.multi_hand_landmarks:  # handLandMarks in this line is single hand
                if draw:
                    # HAND_CONNECTIONS - for connecting dots on hands
                    self.mpDraw.draw_landmarks(img, handLandMarks, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, hand_num=0, draw=True):

        self.lmlist = []  # list for all landmarks position to return

        if self.results.multi_hand_landmarks:
            choose_hand = self.results.multi_hand_landmarks[hand_num]

            for id, lm in enumerate(choose_hand.landmark):
                # print(id, lm)
                height, width, channel = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)  # finding position
                # print(id, cx, cy)
                self.lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 255, 255), cv2.FILLED)

        return self.lmlist

    def fingersUp(self):
        # if len(self.lmlist) != 0:
            fingers = []
            # thumb condition for left hand
            # change greater than for right hand
            # if self.leftHand == True:
            if self.lmlist[self.tipIDs[0]][1] < self.lmlist[self.tipIDs[0] - 1][1]:  # 1 is x-axis
                fingers.append(1)
            else:
                fingers.append(0)
            # elif self.leftHand == False:
            #     if self.lmlist[self.tipIDs[0]][1] > self.lmlist[self.tipIDs[0] - 1][1]:  # 1 is x-axis
            #         fingers.append(1)
            #     else:
            #         fingers.append(0)

            # Fingers
            for id in range(1, 5): # loop for four fingers, thumb not included
                # if lmlist[8][2] < lmlist[6][2]:
                if self.lmlist[self.tipIDs[id]][2] < self.lmlist[self.tipIDs[id] - 2][2]: # 2 is y-axis
                    fingers.append(1)
                else:
                    fingers.append(0)

        # print(fingers)
        # totalFingers = fingers.count(1)
        # print(totalFingers)

            return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmlist[p1][1:]
        x2, y2 = self.lmlist[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]
def main():
    # for frame rate showing setting 0 as initial
    previous_time = 0
    current_time = 0

    capture = cv2.VideoCapture(0)

    detector = handDetector()

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


if __name__ == "__main__":
    main()
