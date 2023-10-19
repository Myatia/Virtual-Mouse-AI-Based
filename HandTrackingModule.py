import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxNumberHands=2, modelComplexity=1, minDetectionConfidence=0.5,
                 minTrackingConfidence=0.5):
        self.mode = mode
        self.maxNumberHands = maxNumberHands
        self.modelComplexity = modelComplexity
        self.minDetectionConfidence = minDetectionConfidence
        self.minTrackingConfidence = minTrackingConfidence

        self.mpHands = mp.solutions.hands  # formalities before using media pipe
        self.hands = self.mpHands.Hands(self.mode, self.maxNumberHands, self.modelComplexity,
                                        self.minDetectionConfidence, self.minTrackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils

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

        lmlist = []  # list for all landmarks position to return

        if self.results.multi_hand_landmarks:
            choose_hand = self.results.multi_hand_landmarks[hand_num]

            for id, lm in enumerate(choose_hand.landmark):
                # print(id, lm)
                height, width, channel = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)  # finding position
                # print(id, cx, cy)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 255, 255), cv2.FILLED)

        return lmlist


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
