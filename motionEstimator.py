import numpy as np
import cv2


## @class MotionEstimator
#
# The class is respnsible, to calculate poses based on the matched feature list
# and the depth map.
# It generates a trajectory based on the output rotation and translation matrices.
# Source: https://github.com/FoamoftheSea/KITTI_visual_odometry
class MotionEstimator():
    def __init__(self):
        self.frameCount = 0
        self.trajectory = list()
        self.trajectory = np.zeros((1, 3, 4))
        self.T_tot = np.eye(4)
        self.trajectory[0] = self.T_tot[:3, :]
        self.skipTraj = False

        self.inliersLimit = 20
        self.maxDepth = 200
        self.reprojectionError = 0.15

    ## @brief Estimate camera motion from a pair of subsequent image frames
    #
    # @param match -- list of matched features from the pair of images
    # @param kp1 -- list of the keypoints in the first image
    # @param kp2 -- list of the keypoints in the second image
    # @param k -- camera intrinsic calibration matrix
    # @param depth1 -- Depth map of the first frame.
    #                  Set to None to use Essential Matrix decomposition
    #
    # @return rmat -- estimated 3x3 rotation matrix
    # @return tvec -- estimated 3x1 translation vector
    # @return image1_points -- matched feature pixel coordinates in the first image.
    #                 image1_points[i] = [u, v] -> pixel coordinates of i-th match
    # @return image2_points -- matched feature pixel coordinates in the second image.
    #                image2_points[i] = [u, v] -> pixel coordinates of i-th match
    def estimate_motion(self, match, kp1, kp2, k, depth1):
        rmat = np.eye(3)
        tvec = np.zeros((3, 1))

        image1_points = np.float32([kp1[m.queryIdx].pt for m in match])
        image2_points = np.float32([kp2[m.trainIdx].pt for m in match])

        cx = k[0, 2]
        cy = k[1, 2]
        fx = k[0, 0]
        fy = k[1, 1]
        object_points = np.zeros((0, 3))
        delete = []

        # Extract depth information of query image
        # at match points and build 3D positions.
        for i, (u, v) in enumerate(image1_points):
            z = depth1[int(v), int(u)]

            # If the depth at the position of our matched feature is above 3000,
            # then we ignore this feature because we don't actually know the depth
            # and it will throw our calculations off.
            # We add its index to a list of coordinates to delete from our
            # keypoint lists, and continue the loop. After the loop,
            # we remove these indices.
            if z > self.maxDepth:
                delete.append(i)
                continue

            # Use arithmetic to extract x and y (faster than using inverse of k)
            x = z * (u - cx) / fx
            y = z * (v - cy) / fy

            object_points = np.vstack([object_points, np.array([x, y, z])])

        image1_points = np.delete(image1_points, delete, 0)
        image2_points = np.delete(image2_points, delete, 0)

        # Use PnP algorithm with RANSAC for robustness to outliers
        try:
            _, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_points,
                image2_points,
                k,
                None,
                reprojectionError=self.reprojectionError)
            if inliers is not None:
                print('Number of inliers: {}/{} matched features'.format(
                    len(inliers), len(match)))

            # Only accept the result if there are certain number of inliers.
            if len(inliers) < self.inliersLimit:
                return None, None, None, None
        except Exception as e:
            print(e)
            print("Not enough feature matches to estimate motion, skipping this step.")
            return None, None, None, None
        # Above function returns axis angle rotation representation rvec,
        # use Rodrigues formula to convert this to our desired format
        # of a 3x3 rotation matrix
        rmat = cv2.Rodrigues(rvec)[0]

        return rmat, tvec, image1_points, image2_points

    ## @brief Function to perform visual odometry.
    #
    # Takes as input a Data_Handler object and optional parameters.
    #
    # @param match -- list of matched features from the pair of images
    # @param kp1 -- list of the keypoints in the first image
    # @param kp2 -- list of the keypoints in the second image
    # @param k_left -- left camera intrinsic calibration matrix
    # @param depth -- Depth map of the first frame.
    #                  Set to None to use Essential Matrix decomposition
    #
    # @return trajectory -- Array of shape Nx3x4 of estimated poses of vehicle
    #                       for each computed frame.
    def calculate_trajectory(
            self,
            matches,
            kp0,
            kp1,
            k_left,
            depth):
        # Estimate motion between sequential images of the left camera
        rmat, tvec, img1_points, img2_points = self.estimate_motion(
            matches, kp0, kp1, k_left, depth)
        if rmat is None:
            return self.trajectory

        # Create blank homogeneous transformation matrix
        Tmat = np.eye(4)
        # Place resulting rotation matrix  and translation vector
        # in their proper locations in homogeneous T matrix
        Tmat[:3, :3] = rmat
        Tmat[:3, 3] = tvec.T

        # The SolvePnPRansac() function computes a pose that relates points in the global
        # coordinate frame to the camera's pose.
        # We used the camera's pose in the first image as the global coordinate frame,
        # reconstruct 3D positions of the features in the image
        # using stereo depth estimation, then find a pose which relates the camera in
        # the next frame to those 3D points. When tracking the vehicle pose over time,
        # what we actually want is to relate the points in the camera's coordinate
        # frame to the global frame, so we want the opposite (inverse) of the
        # transformation matrix provided to us by the SolvePnPRansac function.
        # Recall from the earlier discussion that we can find the inverse of
        # transformation matrix by making it homogeneous by adding a row of (0, 0, 0, 1)
        # to it, then taking its inverse. Further, we are tracking the vehicle
        # motion from the very first camera pose, so we need the cumulative product of the
        # inverses of each estimated camera pose given to us by SolvePnPRansac.
        # Thus, below we iteratively multiply the T_tot homogeneous transformation matrix
        # that we instantiated before the for loop by the inverse
        # of each successive pose we estimate, and save its current values
        # into our estimated poses at an index corresponding to our current frame.
        # This way, the transformation matrix at each index will be one that relates
        # 3D homogeneous coordinates in the camera's frame to the global coordinate
        # frame, which is the coordinate frame of the camera's first position.
        # The translation vector component of this transformation matrix will describe
        # where the camera's curent
        # origin is in this global referece frame.
        T_tot = self.T_tot.dot(np.linalg.inv(Tmat))

        # Place pose estimate in i+1 to correspond to the second image,
        # which we estimated for
        self.trajectory = np.append(self.trajectory, np.zeros((1, 3, 4)), axis=0)
        self.trajectory[self.frameCount + 1, :, :] = T_tot[:3, :]
        self.T_tot = T_tot
        self.frameCount += 1

        return self.trajectory
