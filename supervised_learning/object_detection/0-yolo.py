#!/usr/bin/env python3
"""Module for YOLO v3 object detection algorithm."""
import numpy as np
from tensorflow import keras


class Yolo:
    """Class that uses the YOLO v3 algorithm to perform object detection."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize the Yolo object detection model.

        Args:
            model_path (str): Path to the Darknet Keras model file.
            classes_path (str): Path to the file containing class names,
                listed in order of index.
            class_t (float): Box score threshold for the initial
                filtering step.
            nms_t (float): IOU threshold for non-max suppression.
            anchors (numpy.ndarray): Array of shape (outputs, anchor_boxes, 2)
                containing all anchor boxes, where outputs is the number of
                model outputs, anchor_boxes is the number of anchor boxes
                per output, and 2 represents [anchor_box_width,
                anchor_box_height].
        """
        self.model = keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
