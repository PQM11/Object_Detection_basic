import datetime
import cv2

import numpy as np
import time

np.random.seed(20)
class Detector:
    FRAMES_TO_BANISH = 10 # Tiempo máximo que persisten los resultados.
    FRAMES_TO_ANALYSIS = 2 # De cada X frames de vídeo, busco naranjas en 1.
    THRESHOLD = 0.35 # límite (por debajo) de confianza requerido para mostrar resultado.
    
    def __init__(self, videoPath, configPath, modelPath, classesPath):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath

        ###############################################

        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320,320)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        self.readClasses()

    def readClasses(self):
        with open(self.classesPath, 'r') as f:
            self.classesList = f.read().splitlines()

        self.classesList.insert(0,'__Background__')
        
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList),3))
        # print(self.classesList)

    def onVideo(self):
        """
        It reads a video, and every FRAMES_TO_ANALYSIS frames it detects oranges and draws a rectangle
        around them
        :return: the following:
        """
        cap = cv2.VideoCapture(self.videoPath)
        current_frame = 0
        startTime = 0
        bboxIdx = []
        confidences = []
        bboxs = []
        confidences = []
        classLabelIDs = None
        last_setup = None
        classLabelIDs = []
        fps=0
        startTime =  time.time()
        
        if (cap.isOpened()==False):
            print("Error, opening file...")
            return 
        (success,image) = cap.read()

        while success:
            #fps = 1/(currentTime - startTime)
            # print(fps)
            if current_frame % self.FRAMES_TO_ANALYSIS == 0 : # Si me toca analizar
                currentTime = time.time()
                fps =  self.FRAMES_TO_ANALYSIS /(currentTime - startTime) 
                startTime = currentTime
                classLabelIDs_new, confidences_new, bboxs_new = self.net.detect(image, confThreshold = self.THRESHOLD)

                bboxs_new = list(bboxs_new)
                confidences_new = list(np.array(confidences_new).reshape(1,-1)[0])
                confidences_new = list(map(float, confidences_new))

                #nms_threshold controla el número de recuadros que recoge la imagen
                #cualquier cuadro delimitador que tenga una superposición superior a 20 sería eliminado
                bboxIdx_new = cv2.dnn.NMSBoxes(bboxs_new, confidences_new, score_threshold = self.THRESHOLD, nms_threshold = 0.9)
                
                if len(bboxIdx_new) > len(bboxIdx): # Si tengo más naranjas ahora que antes, las muestro
                    classLabelIDs = classLabelIDs_new
                    bboxs = bboxs_new
                    bboxIdx = bboxIdx_new
                    confidences = confidences_new
                    last_setup = current_frame
                elif (current_frame - last_setup) >= self.FRAMES_TO_BANISH: # Si ha pasado más de FRAMES TO BANISH pues también muestro lo nuevo.
                    classLabelIDs = classLabelIDs_new
                    bboxs = bboxs_new
                    bboxIdx = bboxIdx_new
                    confidences = confidences_new
                    last_setup = current_frame
            no_oranges = 0
            if len(bboxIdx) != 0:
                for i in range(0, len(bboxIdx)-1):
                    bbox = bboxs[np.squeeze(bboxIdx[i])]
                    classConfidence = confidences[np.squeeze(bboxIdx[i])]
                    classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIdx[i])])
                    classLabel = self.classesList[classLabelID]
                    #classColor = [int(c) for c in self.colorList[int(classConfidence*10)]]
                    if (classConfidence) >= 0.5:
                        classColor = [25, 179, 104] # ! OJO : BGR NO RGB
                    elif (classConfidence) >= 0.4:
                        classColor = [87, 250, 171]
                    elif (classConfidence) < 0.4:
                        classColor = [196, 254, 226]
                    
                    if classLabel != "orange":
                        no_oranges += 1
                        continue
                    else:
                        classLabel = "naranja"
                        # print(classLabel)
                        displayText = "{}:{:.2f}".format(classLabel,classConfidence)
                        x,y,w,h = bbox

                        cv2.rectangle(image, (x,y), (x+h, y+h), color = classColor, thickness=1)
                        cv2.putText(image, displayText, (x,y-10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)
                        #print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
                        #print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

                        ##########################################

                        lineWidth = min(int(w * 0.3), int(h * 0.3))

                        cv2.line(image, (x,y), (x + lineWidth, y), classColor, thickness= 5)
                        cv2.line(image, (x,y), (x, y + lineWidth), classColor, thickness= 5)

                        cv2.line(image, (x+w ,y), (x + w - lineWidth, y), classColor, thickness= 5)
                        cv2.line(image, (x+w ,y), (x + w, y + lineWidth), classColor, thickness= 5)

                        ##########################################

                        cv2.line(image, (x,y + h), (x + lineWidth, y + h), classColor, thickness= 5)
                        cv2.line(image, (x,y + h), (x, y + h - lineWidth), classColor, thickness= 5)

                        cv2.line(image, (x+w ,y + h), (x + w - lineWidth, y + h), classColor, thickness= 5)
                        cv2.line(image, (x+w ,y + h), (x + w, y + h - lineWidth), classColor, thickness= 5)
            
            current_frame = current_frame+1
            cv2.putText(image, "FPS: " + str(int(fps)), (20,70), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
            cv2.putText(image, "COUNTER: " + str(len(bboxIdx) - no_oranges), (20,40), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
            cv2.imshow("Result", image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            if key == ord(' '):
                while 1:
                    key = cv2.waitKey(1) & 0xFF
                    if key== ord(' '):
                        break
                    
            (success, image) = cap.read()
        cv2.destroyAllWindows()

