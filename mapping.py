import numpy as np


## @class Landmark Contains the landmark details
#
# A landmark is distinctive point, identified on both left and right image
# It is used to create mapping.
# At the moment it is very plain and just stores the absolute coordinates.
class Landmark():
    def __init__(
            self,
            landmark_id,
            points_3D):
        self.landmark_id = landmark_id
        self.pt_3d = points_3D  # 3d position in the world frame


## @class FeatureMap
#
# Feature map contains all mapping related information.
# At the moment it just stores the list of landmarks.
class FeatureMap():
    def __init__(self):
        self.curr_landmark_id = 0
        self.landmarks = dict()

    # Sets up the landmark coordinate in absolute coordinates.
    def init_landmarks(
            self,
            features3d,
            T_c_w):
        for i in range(0, len(features3d)):
            abs_kp = np.matmul(
                T_c_w,
                np.array([features3d[i][0], features3d[i][1], features3d[i][2], 1.0]))
            # create a landmark
            landmark_to_add = Landmark(
                self.curr_landmark_id,
                abs_kp)
            self.curr_landmark_id += 1
            # insert the landmark
            self.landmarks[landmark_to_add.landmark_id] = landmark_to_add

    # Returns the landmark point cloud.
    def get_pointcloud(self):
        pointcloud = np.empty([len(self.landmarks), 3])
        for id, landmark in self.landmarks.items():
            pointcloud[id] = [landmark.pt_3d[0], -landmark.pt_3d[1], -landmark.pt_3d[2]]
        return pointcloud
