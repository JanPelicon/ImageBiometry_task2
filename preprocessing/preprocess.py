import cv2
import numpy as np


class Preprocess:

    @staticmethod
    def histogram_equalization_rgb(image):
        intensity_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        intensity_img[:, :, 0] = cv2.equalizeHist(intensity_img[:, :, 0])
        return cv2.cvtColor(intensity_img, cv2.COLOR_YCrCb2BGR)

    @staticmethod
    def gamma_correction(image, gamma=1.0):
        table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    @staticmethod
    def sharpen(image):
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        return cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
