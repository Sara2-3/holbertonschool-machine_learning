#!/usr/bin/env python3
"""Module for YOLO v3 object detection algorithm."""
import glob
import os
import cv2
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
                - filtered_boxes: numpy.ndarray of shape (?, 4).
                - box_classes: numpy.ndarray of shape (?,).
                - box_scores: numpy.ndarray of shape (?).
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            scores = box_confidences[i] * box_class_probs[i]
            best_class = np.argmax(scores, axis=-1)
            best_score = np.max(scores, axis=-1)
            mask = best_score >= self.class_t
            filtered_boxes.append(boxes[i][mask])
            box_classes.append(best_class[mask])
            box_scores.append(best_score[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Apply non-max suppression to eliminate redundant overlapping boxes.

        Args:
            filtered_boxes (numpy.ndarray): Shape (?, 4) containing all
                filtered bounding boxes.
            box_classes (numpy.ndarray): Shape (?,) containing the class
                number for each filtered box.
            box_scores (numpy.ndarray): Shape (?,) containing the box score
                for each filtered box.

        Returns:
            tuple: (box_predictions, predicted_box_classes,
                predicted_box_scores)
                - box_predictions: numpy.ndarray of shape (?, 4).
                - predicted_box_classes: numpy.ndarray of shape (?,).
                - predicted_box_scores: numpy.ndarray of shape (?,).
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            cls_mask = box_classes == cls
            cls_boxes = filtered_boxes[cls_mask]
            cls_scores = box_scores[cls_mask]

            order = np.argsort(cls_scores)[::-1]
            cls_boxes = cls_boxes[order]
            cls_scores = cls_scores[order]

            while len(cls_boxes) > 0:
                box_predictions.append(cls_boxes[0])
                predicted_box_classes.append(cls)
                predicted_box_scores.append(cls_scores[0])

                if len(cls_boxes) == 1:
                    break

                x1 = np.maximum(cls_boxes[0, 0], cls_boxes[1:, 0])
                y1 = np.maximum(cls_boxes[0, 1], cls_boxes[1:, 1])
                x2 = np.minimum(cls_boxes[0, 2], cls_boxes[1:, 2])
                y2 = np.minimum(cls_boxes[0, 3], cls_boxes[1:, 3])

                inter_w = np.maximum(0, x2 - x1)
                inter_h = np.maximum(0, y2 - y1)
                intersection = inter_w * inter_h

                area_best = ((cls_boxes[0, 2] - cls_boxes[0, 0]) *
                             (cls_boxes[0, 3] - cls_boxes[0, 1]))
                areas_rest = ((cls_boxes[1:, 2] - cls_boxes[1:, 0]) *
                              (cls_boxes[1:, 3] - cls_boxes[1:, 1]))
                union = area_best + areas_rest - intersection

                iou = intersection / union

                keep = np.where(iou < self.nms_t)[0]
                cls_boxes = cls_boxes[keep + 1]
                cls_scores = cls_scores[keep + 1]

        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """Load images from a folder.

        Args:
            folder_path (str): Path to the folder holding all images to load.

        Returns:
            tuple: (images, image_paths)
                - images: list of images as numpy.ndarrays.
                - image_paths: list of paths to the individual images.
        """
        image_paths = glob.glob(folder_path + '/*')
        images = [cv2.imread(path) for path in image_paths]

        return images, image_paths

    def preprocess_images(self, images):
        """Preprocess images for the Darknet model.

        Args:
            images (list): List of images as numpy.ndarrays.

        Returns:
            tuple: (pimages, image_shapes)
                - pimages: numpy.ndarray of shape (ni, input_h, input_w, 3)
                  containing all preprocessed images, rescaled to [0, 1].
                - image_shapes: numpy.ndarray of shape (ni, 2) containing
                  the original height and width of each image.
        """
        input_w = self.model.input_shape[1]
        input_h = self.model.input_shape[2]

        pimages = []
        image_shapes = []

        for image in images:
            image_shapes.append(image.shape[:2])

            resized = cv2.resize(
                image,
                (input_w, input_h),
                interpolation=cv2.INTER_CUBIC
            )

            pimages.append(resized / 255.0)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """Display image with bounding boxes, class names, and box scores.

        Args:
            image (numpy.ndarray): Unprocessed image.
            boxes (numpy.ndarray): Boundary boxes for the image.
            box_classes (numpy.ndarray): Class indices for each box.
            box_scores (numpy.ndarray): Box scores for each box.
            file_name (str): File path where the original image is stored.
        """
        for i, box in enumerate(boxes):
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            class_name = self.class_names[box_classes[i]]
            score = round(float(box_scores[i]), 2)
            label = "{} {:.2f}".format(class_name, score)

            cv2.putText(
                image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

        cv2.imshow(file_name, image)
        key = cv2.waitKey(0)

        if key == ord('s'):
            os.makedirs('detections', exist_ok=True)
            cv2.imwrite(os.path.join('detections', file_name), image)

        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """Run full YOLO prediction pipeline on all images in a folder.

        Args:
            folder_path (str): Path to the folder holding all images
                to predict.

        Returns:
            tuple: (predictions, image_paths)
                - predictions: list of tuples (boxes, box_classes, box_scores)
                  for each image.
                - image_paths: list of image paths corresponding to each
                  prediction in predictions.
        """
        images, image_paths = self.load_images(folder_path)
        pimages, image_shapes = self.preprocess_images(images)

        # Run model on all preprocessed images at once
        outputs = self.model.predict(pimages)

        # model returns list of outputs; ensure it is always a list
        if not isinstance(outputs, list):
            outputs = [outputs]

        predictions = []

        for i, image in enumerate(images):
            # Collect this image's outputs across all scales
            image_outputs = [output[i] for output in outputs]

            boxes, box_confidences, box_class_probs = self.process_outputs(
                image_outputs, image_shapes[i]
            )

            boxes, box_classes, box_scores = self.filter_boxes(
                boxes, box_confidences, box_class_probs
            )

            boxes, box_classes, box_scores = self.non_max_suppression(
                boxes, box_classes, box_scores
            )

            # Window name is filename only, without full path
            file_name = os.path.basename(image_paths[i])

            self.show_boxes(image, boxes, box_classes, box_scores, file_name)

            predictions.append((boxes, box_classes, box_scores))

        return predictions, image_paths
