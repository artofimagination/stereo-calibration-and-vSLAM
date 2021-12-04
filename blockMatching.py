import cv2
import numpy as np


REMAP_INTERPOLATION = cv2.INTER_LINEAR


# Calculates the epipolar lines for visualization purposes.
def calculateEpipolarLine(frame):
    # sift = cv2.SIFT() causes crash during calling detectAndCompute
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(frame.leftImage, None)
    kp2, des2 = sift.detectAndCompute(frame.rightImage, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    # We select only inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    # Find epilines corresponding to points in right image (second image)
    # and drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawLines(
        frame.leftImage, frame.rightImage, lines1, pts1, pts2)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawLines(
        frame.rightImage, frame.leftImage, lines2, pts2, pts1)
    return frame


# Undistorts the camera images, using the calibration data.
def undistort(calibration, frame):
    leftMapX = calibration["leftMapX"]
    leftMapY = calibration["leftMapY"]
    rightMapX = calibration["rightMapX"]
    rightMapY = calibration["rightMapY"]
    frame.leftImage = cv2.remap(
        frame.leftImage,
        leftMapX,
        leftMapY,
        REMAP_INTERPOLATION)
    frame.rightImage = cv2.remap(
        frame.rightImage,
        rightMapX,
        rightMapY,
        REMAP_INTERPOLATION)
    return frame


# Decomposes projection matrix into intrinsic, rotation and 3D translation matrices.
def decompose_projection_matrix(p):
    '''
    Shortcut to use cv2.decomposeProjectionMatrix(),
    which only returns k, r, t, and divides
    t by the scale, then returns it as a vector with shape (3,) (non-homogeneous)

    Arguments:
    p -- projection matrix to be decomposed

    Returns:
    k, r, t -- intrinsic matrix, rotation matrix, and 3D translation vector

    '''
    k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(p)
    t = (t / t[3])[:3]

    return k, r, t


# Draws the specified lines on the input images.
def drawLines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c, _ = img1.shape
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


# Generates mask based on the depth map. The mask has zero values on the left side
# with a size equal to the blank area resulting of disparity calculation.
def generateDepthMapMask(depthImage, leftImage):
    firstNonMaxIndex = 0
    for i, pixel in enumerate(depthImage[4]):
        if pixel < depthImage.max():
            firstNonMaxIndex = i
            break
    mask = np.zeros(depthImage.shape[:2], dtype=np.uint8)
    ymax = depthImage.shape[0]
    xmax = depthImage.shape[1]
    cv2.rectangle(mask, (firstNonMaxIndex, 0), (xmax, ymax), (255), thickness=-1)
    return mask


## @class BlockMatching
#  @brief Responsible to generate the depth map using block matching
#
#  The algorithm can be either block matching or semi global block matching.
class BlockMatching():

    def __init__(self):
        # Stereo block matching parameters
        self.bmType = 0
        self.min_disp = 0
        self.num_disp = 128
        self.blockSize = 5
        self.uniquenessRatio = 1
        self.speckleWindowSize = 2
        self.speckleRange = 80
        self.disp12MaxDiff = 180
        self.preFilterCap = 63
        self.preFilterSize = 5
        self.preFilterType = 0
        self.smallerBlockSize = 0
        self.textureThreshold = 0

        self.drawEpipolar = False

    # Generates depth map using block matching, or semi global block matching.
    def createDepthMap(self, calibration, frame):
        imageSize = tuple(calibration["imageSize"])
        leftROI = tuple(calibration["leftROI"])
        rightROI = tuple(calibration["rightROI"])

        # TODO: Why these values in particular?
        # TODO: Try applying brightness/contrast/gamma
        # adjustments to the images
        if self.bmType == 0:
            stereoMatcher = cv2.StereoBM_create()
            stereoMatcher.setROI1(leftROI)
            stereoMatcher.setROI2(rightROI)
            stereoMatcher.setPreFilterType(self.preFilterType)
            stereoMatcher.setPreFilterSize(self.preFilterSize)
            stereoMatcher.setPreFilterCap(self.preFilterCap)
            stereoMatcher.setSmallerBlockSize(self.smallerBlockSize)
            stereoMatcher.setTextureThreshold(self.textureThreshold)
        else:
            stereoMatcher = cv2.StereoSGBM_create()
            stereoMatcher.setP1(8 * 1 * self.blockSize ** 2)
            stereoMatcher.setP2(32 * 1 * self.blockSize ** 2)
            stereoMatcher.setMode(cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        stereoMatcher.setMinDisparity(self.min_disp)
        stereoMatcher.setNumDisparities(self.num_disp)
        stereoMatcher.setBlockSize(self.blockSize)

        stereoMatcher.setSpeckleRange(self.speckleRange)
        stereoMatcher.setSpeckleWindowSize(self.speckleWindowSize)
        stereoMatcher.setDisp12MaxDiff(self.disp12MaxDiff)
        stereoMatcher.setUniquenessRatio(self.uniquenessRatio)

        leftHeight, leftWidth = frame.leftImage.shape[:2]
        rightHeight, rightWidth = frame.rightImage.shape[:2]
        if (leftWidth, leftHeight) != imageSize:
            print("Left camera has different size than the calibration data")
            return (frame, None)

        if (rightWidth, rightHeight) != imageSize:
            print("Right camera has different size than the calibration data")
            return (frame, None)

        frame = undistort(calibration, frame)

        grayLeft = cv2.cvtColor(frame.leftImage, cv2.COLOR_BGR2GRAY)
        grayRight = cv2.cvtColor(frame.rightImage, cv2.COLOR_BGR2GRAY)

        if self.drawEpipolar is True:
            frame = calculateEpipolarLine(frame)

        depth = stereoMatcher.compute(grayLeft, grayRight)
        # Adding some smoothing to increase image quality.
        depth = cv2.medianBlur(depth, 5).astype(np.float32) / 16
        k_left, _, t_left = decompose_projection_matrix(calibration["leftProjection"])
        _, _, t_right = decompose_projection_matrix(calibration["rightProjection"])
        f = k_left[0][0]

        # Calculate baseline of stereo pair
        b = t_right[0] - t_left[0]

        # Avoid instability and division by zero
        depth[depth == 0.0] = 0.1
        depth[depth == -1.0] = 0.1

        # Make empty depth map then fill with depth
        depth_map = np.ones(depth.shape)
        depth_map = f * b / depth

        return frame, depth_map
