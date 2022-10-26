from detector import *
import os

def main():
    videoPath = "test_videos/vid_1.MOV"
    # videoPath = 'test_videos/VIDEO 1 - ARBOL 1 FOTO 1.jpg' #IMage
    # videoPath = 0

    configPath = os.path.join("model_data", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath = os.path.join("model_data", "frozen_inference_graph.pb")
    classesPath = os.path.join("model_data", "coco.names")

    detector = Detector(videoPath, configPath, modelPath, classesPath)
    detector.onVideo()

if __name__ == '__main__':
    main()
