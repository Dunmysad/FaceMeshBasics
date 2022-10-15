import cv2 as cv
import mediapipe as mp
import time


class FaceMashDetector():
    def __init__(self, staticMode=False, maxFaces=2, refine=False, minconfidence=0.5, maxconfidence=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.refine = refine
        self.minconfidence = minconfidence
        self.maxconfidence = maxconfidence


        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, self.refine, self.minconfidence, self.maxconfidence)
        self.drawSpec = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)

    def findFaceMesh(self, img, draw=True):

        self.imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                      self.drawSpec, self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    # print(lm)
                    ih, iw, ic = img.shape
                    x, y = int(lm.x*iw), int(lm.y*ih)
                    # print(id, x, y)
                    # 脸地标编号
                    # cv.putText(img, str(id), (x, y), cv.FONT_HERSHEY_PLAIN, 0.5, (255, 0, 255), 1)
                    face.append([x,y])
                faces.append(face)
        return img, faces



def main():
    # cap = cv.VideoCapture('Video/1-热恋冰激凌- 程Yooooo-1080P 高清-AVC.mp4')
    cap = cv.VideoCapture(0)
    pTime = 0
    detector = FaceMashDetector()

    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)
        if len(faces) != 0:
            print(faces)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText(img, f'FPS:{int(fps)}', (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv.imshow('image', img)
        cv.waitKey(1)



if __name__ == '__main__':
    main()
