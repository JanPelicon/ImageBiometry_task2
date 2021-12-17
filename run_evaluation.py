import signal
import cv2
import numpy as np
import glob
import os
from pathlib import Path
import json
from preprocessing.preprocess import Preprocess
from metrics.evaluation import Evaluation
import time


class EvaluateAll:

    def __init__(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        with open('config.json') as config_file:
            config = json.load(config_file)

        self.images_path = config['images_path']
        self.annotations_path = config['annotations_path']

    @staticmethod
    def get_annotations(annot_name):
        with open(annot_name) as f:
            lines = f.readlines()
            annot = []
            for line in lines:
                l_arr = line.split(" ")[1:5]
                l_arr = [int(i) for i in l_arr]
                annot.append(l_arr)
        return annot

    def run_evaluation(self):

        # net_name = "yolov2-tiny"
        # net_name = "yolov3"
        # net_name = "yolov3-tiny"
        net_name = "yolov4-tiny"
        show = True
        print_output = not True
        net_input_size = (416, 416)
        conf_threshold = 0.4  # 0.5
        nms_threshold = 0.2  # 0.4

        # Change the following detector and/or add your detectors below
        # import detectors.cascade_detector.detector as cascade_detector
        # import detectors.your_super_detector.detector as super_detector
        # cascade_detector = cascade_detector.Detector()

        path_yolo = "E:/sb_project/" + net_name + "/"
        path_weights = path_yolo + net_name + "_ear_final.weights"
        path_config = path_yolo + net_name + "_ear.cfg"
        net_name = "yolov4-tiny"

        preprocess = Preprocess()
        evaluation = Evaluation()
        net = cv2.dnn.readNet(path_weights, path_config)
        output_layers = [net.getLayerNames()[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        image_list = sorted(glob.glob(self.images_path + '/*.png', recursive=True))
        iou_arr = []
        precision_arr = []
        recall_arr = []
        time_arr = []

        for index, image_name in enumerate(image_list):
            if index % 30 == 0:
                print("{}%".format(index/len(image_list)*100))

            # Read an image
            image = cv2.imread(image_name)

            # Apply some preprocessing

            #image = preprocess.histogram_equalization_rgb(image)
            #image = preprocess.sharpen(image)
            #image = preprocess.gamma_correction(image, 0.95)


            # Run the detector. It runs a list of all the detected bounding-boxes.
            # In segmentor you only get a mask matrices, but use the iou_compute in the same way.
            # prediction_list = cascade_detector.detect(img)

            height, width, channels = image.shape
            # 0.00392 = 1 / 255
            image_input = cv2.dnn.blobFromImage(image, scalefactor=0.00392, size=net_input_size, mean=(0, 0, 0),
                                                swapRB=True, crop=False)

            time_start = round(time.time() * 1000)
            net.setInput(image_input)
            outputs = net.forward(output_layers)
            time_process = round(time.time() * 1000) - time_start
            time_arr.append(time_process)

            confidences = []
            boxes = []

            for output in outputs:
                print(output)
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > conf_threshold:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))

            # Read annotations:
            annot_name = os.path.join(self.annotations_path, Path(os.path.basename(image_name)).stem) + '.txt'
            annot_list = self.get_annotations(annot_name)

            if show:
                for annotation in annot_list:
                    x, y, w, h = annotation
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            prediction_list = []
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    if show:
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    prediction_list.append([x, y, w, h])

            if show:
                cv2.imshow("Image", image)
                cv2.waitKey(0)

            # Only for detection:
            p, gt = evaluation.prepare_for_detection(prediction_list, annot_list)

            iou = evaluation.iou_compute(p, gt)
            iou_arr.append(iou)

            precision, recall = evaluation.precision_recall(prediction_list, annot_list, net_input_size)
            precision_arr.append(precision)
            recall_arr.append(recall)

            if print_output:
                print("{}\nannot: {}\npredict: {}\niou: {}".format(image_name, annot_list, prediction_list, iou))

        avg_iou = np.average(iou_arr)
        avg_precision = np.average(precision_arr)
        avg_recall = np.average(recall_arr)
        avg_time = np.average(time_arr)
        print("Average IOU = {:0.3f}".format(avg_iou))
        print("Average precision = {:0.3f}".format(avg_precision))
        print("Average recall = {:0.3f}".format(avg_recall))
        print("Average inference time = {:0.3f} ms".format(avg_time))
        cv2.destroyAllWindows()


if __name__ == '__main__':
    ev = EvaluateAll()
    ev.run_evaluation()
    os.kill(os.getpid(), signal.SIGTERM)
