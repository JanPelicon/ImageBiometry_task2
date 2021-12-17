import cv2
import numpy as np

class Evaluation:

    def convert2mask(self, mt, shape):
        # Converts coordinates of bounding-boxes into blank matrix with values set where bounding-boxes are.

        t = np.zeros([shape, shape])
        for m in mt:
            x, y, w, h = m
            cv2.rectangle(t, (x,y), (x+w, y+h), 1, -1)
        return t

    def prepare_for_detection(self, prediction, ground_truth):
            # For the detection task, convert Bounding-boxes to masked matrices (0 for background, 1 for the target). If you run segmentation, do not run this function

            if len(prediction) == 0:
                return [], []

            # Large enough size for base mask matrices:
            shape = 2*max(np.max(prediction), np.max(ground_truth)) 
            
            p = self.convert2mask(prediction, shape)
            gt = self.convert2mask(ground_truth, shape)

            return p, gt

    def iou_compute(self, p, gt):
            # Computes Intersection Over Union (IOU)
            if len(p) == 0:
                return 0

            intersection = np.logical_and(p, gt)
            union = np.logical_or(p, gt)

            iou = np.sum(intersection) / np.sum(union)

            return iou

    def precision_recall(self, p, gt, size):
        center_p = []
        center_gt = []
        for box in p:
            x, y, w, h = box
            center_x = (x + w/2) / size[0]
            center_y = (y + h/2) / size[1]
            center_p.append(np.array([center_x, center_y]))
        for box in gt:
            x, y, w, h = box
            center_x = (x + w/2) / size[0]
            center_y = (y + h/2) / size[1]
            center_gt.append(np.array([center_x, center_y]))
        correct = 0
        for c_p in center_p:
            for c_gt in center_gt:
                dist = np.linalg.norm(c_p - c_gt)
                if dist < 0.05:
                    correct += 1
                    break
        precision = 0
        recall = 0
        if len(center_p) != 0:
            precision = correct / len(center_p)
        if len(center_gt) != 0:
            recall = correct / len(center_gt)
        return precision, recall

    # Add your own metrics here, such as mAP, class-weighted accuracy, ...
