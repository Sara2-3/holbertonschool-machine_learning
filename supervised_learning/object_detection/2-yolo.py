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

    def process_outputs(self, outputs, image_size):
        """Process the outputs from the Darknet model.

        Args:
            outputs (list): List of numpy.ndarrays containing predictions
                from the Darknet model for a single image. Each output has
                shape (grid_height, grid_width, anchor_boxes, 4+1+classes).
            image_size (numpy.ndarray): Array containing the image's original
                size [image_height, image_width].

        Returns:
            tuple: (boxes, box_confidences, box_class_probs)
                - boxes: list of numpy.ndarrays of shape
                  (grid_height, grid_width, anchor_boxes, 4) with processed
                  boundary boxes (x1, y1, x2, y2) relative to original image.
                - box_confidences: list of numpy.ndarrays of shape
                  (grid_height, grid_width, anchor_boxes, 1).
                - box_class_probs: list of numpy.ndarrays of shape
                  (grid_height, grid_width, anchor_boxes, classes).
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size
        input_w = self.model.input_shape[1]
        input_h = self.model.input_shape[2]

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            t_xy = output[:, :, :, :2]
            b_xy = 1 / (1 + np.exp(-t_xy))

            cx = np.arange(grid_width).reshape(1, grid_width, 1)
            cy = np.arange(grid_height).reshape(grid_height, 1, 1)

            b_xy[:, :, :, 0] = (b_xy[:, :, :, 0] + cx) / grid_width
            b_xy[:, :, :, 1] = (b_xy[:, :, :, 1] + cy) / grid_height

            t_wh = output[:, :, :, 2:4]
            anchors_wh = self.anchors[i].reshape(1, 1, anchor_boxes, 2)

            b_wh = np.exp(t_wh) * anchors_wh
            b_wh[:, :, :, 0] /= input_w
            b_wh[:, :, :, 1] /= input_h

            x1 = (b_xy[:, :, :, 0] - b_wh[:, :, :, 0] / 2) * image_width
            y1 = (b_xy[:, :, :, 1] - b_wh[:, :, :, 1] / 2) * image_height
            x2 = (b_xy[:, :, :, 0] + b_wh[:, :, :, 0] / 2) * image_width
            y2 = (b_xy[:, :, :, 1] + b_wh[:, :, :, 1] / 2) * image_height

            box = np.stack([x1, y1, x2, y2], axis=-1)
            boxes.append(box)

            box_conf = 1 / (1 + np.exp(-output[:, :, :, 4:5]))
            box_confidences.append(box_conf)

            box_cls_prob = 1 / (1 + np.exp(-output[:, :, :, 5:]))
            box_class_probs.append(box_cls_prob)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter bounding boxes based on class score threshold.

        Args:
            boxes (list): numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, 4) containing
                processed boundary boxes for each output.
            box_confidences (list): numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, 1) containing
                processed box confidences for each output.
            box_class_probs (list): numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, classes) containing
                processed box class probabilities for each output.

        Returns:
            tuple: (filtered_boxes, box_classes, box_scores)
                - filtered_boxes: numpy.ndarray of shape (?, 4) with all
                  filtered bounding boxes.
                - box_classes: numpy.ndarray of shape (?,) with class number
                  each filtered box predicts.
                - box_scores: numpy.ndarray of shape (?) with box score for
                  each filtered box.
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            # box_score = confidence * class_probability
            # shape: (grid_h, grid_w, anchor_boxes, classes)
            scores = box_confidences[i] * box_class_probs[i]

            # Best class and its score per box
            # shape: (grid_h, grid_w, anchor_boxes)
            best_class = np.argmax(scores, axis=-1)
            best_score = np.max(scores, axis=-1)

            # Mask: keep only boxes with score >= class_t
            mask = best_score >= self.class_t

            filtered_boxes.append(boxes[i][mask])
            box_classes.append(best_class[mask])
            box_scores.append(best_score[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores
