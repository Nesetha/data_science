import cv2
import numpy as np


class ColorDetector:

    def detect_color(self, person_crop):

        if person_crop.size == 0:
            return "Unknown"

        hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)

        avg_hue = np.mean(hsv[:, :, 0])
        avg_sat = np.mean(hsv[:, :, 1])
        avg_val = np.mean(hsv[:, :, 2])

        # Black
        if avg_val < 50:
            return "Black"

        # White
        if avg_sat < 40 and avg_val > 180:
            return "White"

        # Red
        if avg_hue < 10 or avg_hue > 170:
            return "Red"

        # Orange
        if avg_hue < 25:
            return "Orange"

        # Yellow
        if avg_hue < 35:
            return "Yellow"

        # Green
        if avg_hue < 85:
            return "Green"

        # Blue
        if avg_hue < 130:
            return "Blue"

        # Purple
        if avg_hue < 160:
            return "Purple"

        return "Unknown"