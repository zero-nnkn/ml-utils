from abc import ABC, abstractmethod

import cv2
import numpy as np


class OpenPosePredictor(ABC):
    """
    The Pose Estimation class to predict pose keypoints of an image by OpenPose using OpenCV

    Usage: Specify the subclass corresponding to the model you want to use
      COCO:   18 points
      MPI:    15 points
      Body25: 25 points
    """

    def __init__(
        self,
        prototxt_path: str,
        weight_path: str,
        in_size: tuple = (368, 368),
        threshold: float = 0.05,
    ) -> None:
        """
        It initializes the model.

        Args:
          prototxt_path (str): The path to the prototxt file.
          weight_path (str): The path to the caffemodel file.
          in_size (tuple): tuple = (368, 368): the input image dimensions
          threshold (float): float = 0.05: confidence threshold to identify key points
        """
        self.in_size = in_size  # (in_width, in_height)
        self.threshold = threshold
        self._init__model_infor()
        self.pose_net = self._create_model(prototxt_path, weight_path)

    @abstractmethod
    def _init__model_infor(self) -> None:
        """
        > This function initializes the model information
        and will be implemented in subclass corresponding to the model
        """
        raise NotImplementedError

    def _create_model(self, prototxt_path: str, weight_path: str) -> cv2.dnn.Net:
        """
        `_create_model` creates a Caffe model from a prototxt file and a caffe weight file.

        Args:
          prototxt_path (str): The path to the prototxt file.
            Download: https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/models/pose
          weight_path (str): The path to the Caffe model file.
            COCO:    https://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel
            MPI:     https://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel
            BODY_25: https://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/body_25/pose_iter_584000.caffemodel

        Returns:
          The model is being returned.
        """  # noqa
        coco_model = cv2.dnn.readNetFromCaffe(prototxt_path, weight_path)
        return coco_model

    def predict(self, img) -> list:
        """
        It takes an image as input, runs it through the network, and returns a list of keypoints

        Args:
          img: The image to be processed.

        Returns:
          A list of keypoints
        """
        img_h, img_w, _ = img.shape
        self._prepare_input(img)
        output = self.pose_net.forward()
        keypoints = self._parse_keypoints(output, img_size=(img_w, img_h))
        return keypoints

    def _prepare_input(self, img: np.array) -> None:
        """
        It takes an image, resizes it to the input size of the network,
        and sets the image as the input to the network

        Args:
          img (np.array): the input image
        """
        inp_blob = cv2.dnn.blobFromImage(
            img, 1.0 / 255, self.in_size, (0, 0, 0), swapRB=False, crop=False
        )
        self.pose_net.setInput(inp_blob)

    def _parse_keypoints(self, net_ouput: cv2.Mat, img_size: tuple) -> list:
        """
        It takes the output of the neural network, and returns a list of the x,y coordinates of the
        keypoints, and the confidence of each keypoint.

        Args:
          net_ouput (cv2.Mat): the output of the neural network (4D matrix).
            - 1st dim: image ID (if pass more than 1 image)
            - 2nd dim: index of a keypoint.
                Ex COCO:  18 keypoint confidence maps + 1 background + 19*2 Part Affinity Maps
            - 3rd, 4th: height, width of output map
          img_size (tuple): The size of the image that we're going to be processing.

        Returns:
          a list of the x, y, and probability of each keypoint.
        """
        H = net_ouput.shape[2]
        W = net_ouput.shape[3]

        points = []
        for idx in range(len(self.BODY_PARTS)):
            probMap = net_ouput[0, idx, :, :]  # confidence map.

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            # Scale the point to fit on the original image
            x = (img_size[0] * point[0]) / W
            y = (img_size[1] * point[1]) / H

            if prob > self.threshold:
                points.append((x, y, prob))
            else:
                points.append(None)

        return points

    def visualize(self, keypoints: list, img, put_text: bool = False):
        """
        It takes a list of keypoints and an image, and draws the skeleton keypoints on the image.

        Args:
          keypoints (list): list of keypoints
          img: The image to draw the skeleton on
          put_text (bool): If True, the keypoints will be labeled with their index

        Returns:
          The image with the skeleton drawn on it.
        """
        img_copy = img.copy()

        # Draw keypoints
        img_copy = self._visualize_points(keypoints, img, put_text)

        # Draw skeleton
        for pair in self.POSE_PAIRS:
            p1, p2 = self.BODY_PARTS[pair[0]], self.BODY_PARTS[pair[1]]

            if keypoints[p1] and keypoints[p2]:
                x1, y1, prob1 = keypoints[p1]
                x2, y2, prob2 = keypoints[p2]
                cv2.line(
                    img_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3
                )

        return img_copy

    def _visualize_points(self, keypoints: list, img, put_text: bool = False):
        """
        It takes a list of keypoints and an image, and draws a circle around each keypoint
        """
        for i in range(len(self.BODY_PARTS)):
            if keypoints[i]:
                x, y, prob = keypoints[i]
                cv2.circle(
                    img,
                    (int(x), int(y)),
                    15,
                    (0, 255, 255),
                    thickness=-1,
                    lineType=cv2.FILLED,
                )
                if put_text:
                    cv2.putText(
                        img,
                        f"{i}",
                        (int(x), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.4,
                        (0, 0, 255),
                        3,
                        lineType=cv2.LINE_AA,
                    )

        return img


class COCOOpenPosePredictor(OpenPosePredictor):
    def _init__model_infor(self) -> None:
        self.BODY_PARTS = {
            "Nose": 0,
            "Neck": 1,
            "RShoulder": 2,
            "RElbow": 3,
            "RWrist": 4,
            "LShoulder": 5,
            "LElbow": 6,
            "LWrist": 7,
            "RHip": 8,
            "RKnee": 9,
            "RAnkle": 10,
            "LHip": 11,
            "LKnee": 12,
            "LAnkle": 13,
            "REye": 14,
            "LEye": 15,
            "REar": 16,
            "LEar": 17,
            "Background": 18,
        }
        self.POSE_PAIRS = [
            ["Neck", "RShoulder"],
            ["RShoulder", "RElbow"],
            ["RElbow", "RWrist"],
            ["Neck", "LShoulder"],
            ["LShoulder", "LElbow"],
            ["LElbow", "LWrist"],
            ["Neck", "RHip"],
            ["RHip", "RKnee"],
            ["RKnee", "RAnkle"],
            ["Neck", "LHip"],
            ["LHip", "LKnee"],
            ["LKnee", "LAnkle"],
            ["Neck", "Nose"],
            ["Nose", "REye"],
            ["REye", "REar"],
            ["Nose", "LEye"],
            ["LEye", "LEar"],
        ]


class MPIOpenPosePredictor(OpenPosePredictor):
    def _init__model_infor(self) -> None:
        self.BODY_PARTS = {
            "Head": 0,
            "Neck": 1,
            "RShoulder": 2,
            "RElbow": 3,
            "RWrist": 4,
            "LShoulder": 5,
            "LElbow": 6,
            "LWrist": 7,
            "RHip": 8,
            "RKnee": 9,
            "RAnkle": 10,
            "LHip": 11,
            "LKnee": 12,
            "LAnkle": 13,
            "Chest": 14,
            "Background": 15,
        }
        self.POSE_PAIRS = [
            ["Head", "Neck"],
            ["Neck", "RShoulder"],
            ["RShoulder", "RElbow"],
            ["RElbow", "RWrist"],
            ["Neck", "LShoulder"],
            ["LShoulder", "LElbow"],
            ["LElbow", "LWrist"],
            ["Neck", "Chest"],
            ["Chest", "RHip"],
            ["RHip", "RKnee"],
            ["RKnee", "RAnkle"],
            ["Chest", "LHip"],
            ["LHip", "LKnee"],
            ["LKnee", "LAnkle"],
        ]


class Body25OpenPosePredictor(OpenPosePredictor):
    def _init__model_infor(self) -> None:
        self.BODY_PARTS = {
            "Nose": 0,
            "Neck": 1,
            "RShoulder": 2,
            "RElbow": 3,
            "RWrist": 4,
            "LShoulder": 5,
            "LElbow": 6,
            "LWrist": 7,
            "MidHip": 8,
            "RHip": 9,
            "RKnee": 10,
            "RAnkle": 11,
            "LHip": 12,
            "LKnee": 13,
            "LAnkle": 14,
            "REye": 15,
            "LEye": 16,
            "REar": 17,
            "LEar": 18,
            "LBigToe": 19,
            "LSmallToe": 20,
            "LHeel": 21,
            "RBigToe": 22,
            "RSmallToe": 23,
            "RHeel": 24,
            "Background": 25,
        }
        self.POSE_PAIRS = [
            ["Neck", "Nose"],
            ["Neck", "RShoulder"],
            ["RShoulder", "RElbow"],
            ["RElbow", "RWrist"],
            ["Neck", "LShoulder"],
            ["LShoulder", "LElbow"],
            ["LElbow", "LWrist"],
            ["Nose", "REye"],
            ["REye", "REar"],
            ["Nose", "LEye"],
            ["LEye", "LEar"],
            ["Neck", "MidHip"],
            ["MidHip", "RHip"],
            ["RHip", "RKnee"],
            ["RKnee", "RAnkle"],
            ["RAnkle", "RBigToe"],
            ["RBigToe", "RSmallToe"],
            ["RAnkle", "RHeel"],
            ["MidHip", "LHip"],
            ["LHip", "LKnee"],
            ["LKnee", "LAnkle"],
            ["LAnkle", "LBigToe"],
            ["LBigToe", "LSmallToe"],
            ["LAnkle", "LHeel"],
        ]


def add_keypoints_to_json(self, img: np.array, pose_predictor: OpenPosePredictor):
    """
    It takes an image and a pose predictor, and returns a JSON object with the keypoints of the pose

    Args:
      img (np.array): the image to be processed
      pose_predictor (PosePredictor): PosePredictor object

    Returns:
      A dictionary with the keypoints of the person in the image.
    """
    keypoints = [
        p if p else (0, 0, 0) for p in pose_predictor.predict(img)
    ]  # replace None with [0, 0, 0]
    keypoints = [item in p for p in keypoints for item in p]  # flat list
    pose_json = {"version": 1, "people": [{"pose_keypoints": keypoints}]}

    return pose_json
