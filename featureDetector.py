import cv2

from PyQt5 import QtCore
from PyQt5.QtCore import QObject


## @class FeatureDetector
#  @brief Responsible for extracting and matching features
#
#  Requires two consecutive images frames.
class FeatureDetector(QObject):
    infoStrUpdated = QtCore.pyqtSignal(object)

    def __init__(self):
        super(FeatureDetector, self).__init__()
        self.currentFrame = None
        self.previousFrame = None
        # Enables drawing matches on image pairs.
        self.drawMatches = True
        # Message to display in the UI.
        self.message = ""
        # Maximum allowed relative distance between the best matches, (0.0, 1.0)
        self.matchDistanceThreshold = 0.3
        # Holds the feature detector type string (sift, orb, surf)
        self.featureDetector = "sift"
        # Holds the feature matching algorithm type (Brute force, FLANN).
        self.featureMatcher = "BF"

    ## @brief Find keypoints and descriptors for the image
    #
    #    @param image -- a grayscale image
    #
    #    @return kp -- list of the extracted keypoints (features) in an image
    #    @return des -- list of the keypoint descriptors in an image
    def extract_features(self, image, mask):
        if self.featureDetector == "sift":
            det = cv2.SIFT_create()
        elif self.featureDetector == "orb":
            det = cv2.ORB_create()
        elif self.featureDetector == "surf":
            det = cv2.xfeatures2d.SURF_create()

        kp, des = det.detectAndCompute(image, mask)
        return kp, des

    ## @brief sets the current and previous frames.
    def set_frame(self, frame):
        self.previousFrame = self.currentFrame
        self.currentFrame = frame

    ## @brief Match features from two images
    #
    #  @param des1 -- list of the keypoint descriptors in the first image
    #  @param des2 -- list of the keypoint descriptors in the second image
    #  @param sort -- (bool) whether to sort matches by distance. Default is True
    #  @param k -- (int) number of neighbors to match to each feature.
    #
    #  @return matches -- list of matched features from two images.
    #          Each match[i] is k or less matches for the same query descriptor.
    def match_features(self, des1, des2, sort=True, k=2):
        if self.featureMatcher == 'BF':
            if self.featureDetector == 'sift':
                matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)
            elif self.featureDetector == 'orb':
                matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING2, crossCheck=False)
            else:
                raise Exception("Brute force matcher with surf detector not supported ")
            matches = matcher.knnMatch(des1, des2, k=k)
        elif self.featureMatcher == 'FLANN':
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
            matches = matcher.knnMatch(des1, des2, k=k)

        if sort:
            matches = sorted(matches, key=lambda x: x[0].distance)

        return matches

    ## @brief Filter matched features from two images by distance between the best matches
    #
    # @param match -- list of matched features from two images
    #
    # @return filtered_match -- list of good matches, satisfying the distance threshold
    def filter_matches_distance(self, matches):
        filtered_match = []
        for m, n in matches:
            if m.distance <= self.matchDistanceThreshold * n.distance:
                filtered_match.append(m)

        return filtered_match

    ## @brief Extract features and detects feature matches on consecutuve images.
    def detect_features(self):
        if self.previousFrame is not None:
            kp0, des0 = self.extract_features(self.currentFrame, None)
            kp1, des1 = self.extract_features(self.previousFrame, None)
            matches = self.match_features(des0, des1, sort=True)
            self.message = f"Number of matches before filtering: {len(matches)}\n"
            matches = self.filter_matches_distance(matches)
            self.message += f"Number of matches after filtering: {len(matches)}"
            self.infoStrUpdated.emit(self.message)
            image_matches = self.currentFrame
            if self.drawMatches:
                image_matches = cv2.drawMatches(
                    self.currentFrame,
                    kp0,
                    self.previousFrame,
                    kp1,
                    matches,
                    None,
                    flags=2)
            return (kp0, kp1, matches, image_matches)
        return (None, None, None, None)
