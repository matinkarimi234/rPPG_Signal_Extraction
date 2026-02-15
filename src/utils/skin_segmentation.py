import cv2
import numpy as np

def create_skin_mask(image, lower_hsv=(0, 20, 70), upper_hsv=(20, 255, 255)):
    """
    Return binary mask of skin pixels using HSV range.
    Tune lower/upper for your lighting conditions.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    # Optional morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask