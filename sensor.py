import cv2


## @class Frame
#  Contains the current left and right image.
class Frame():
    def __init__(self, leftImage, rightImage):
        self.leftImage = leftImage
        self.rightImage = rightImage


## @class Sensor
# This class implements the camera sensor handling and image outputing
class Sensor():
    def __init__(self):
        self.running = False

        self.camera_width = 640
        self.camera_height = 480

        # List of video devices found in the OS (/dev/video*).
        self.sensor_indices = list()
        self.sensor_devices = list()
        # Left camera capture instance
        self.leftSensor = None
        # Right camera capture instance
        self.rightSensor = None
        # Left camera index, that refers to the appropriate video device.
        self.leftIndex = 2
        # Right camera index, that refers to the appropriate video device.
        self.rightIndex = 1

    # Detects all available video devices in the OS.
    def detectSensors(self):
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f'Camera index available: {i}')
                self.sensor_indices.append(i)
                self.sensor_devices.append(cap)

        if len(self.sensor_indices) < 2:
            raise Exception("Not all sensors are accessible")

    # Initializes and starts the sensors
    # If there are more than 2 sensors
    # make sure the camera indices belong to the correct sensor.
    def startSensors(self):
        print("Starting sensors...")
        if self.running is False:
            self.leftSensor = self.sensor_devices[self.leftIndex]
            self.rightSensor = self.sensor_devices[self.rightIndex]

            # Increase the resolution
            self.leftSensor.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.leftSensor.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            self.rightSensor.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.rightSensor.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)

            # Use MJPEG to avoid overloading the USB 2.0 bus at this resolution
            self.leftSensor.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.rightSensor.set(
                cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

            frame = self.captureFrame()
            if frame.leftImage is None or frame.rightImage is None:
                raise Exception("No frames, quitting app. Try to start again,\
usually the camera modules are ready by the second attempt")

            self.running = True
            print(f"Resolution is {self.camera_width}x{self.camera_height}")
            print("Sensors running...")
        else:
            print("Sensors already running...")

    # Restart sensors.
    def restartSensors(self):
        print("Restarting sensors...")
        self.running = False
        self.releaseVideoDevices()
        self.detectSensors()
        self.startSensors()

    # Capture frame in an efficient way.
    def captureFrame(self):
        # Grab both frames first,
        # then retrieve to minimize latency between cameras
        if not self.leftSensor.grab() or not self.rightSensor.grab():
            print("No more frames")
            return (None, None)

        _, leftFrame = self.leftSensor.retrieve()
        _, rightFrame = self.rightSensor.retrieve()
        return Frame(leftFrame, rightFrame)

    # Clears the sensor devices data
    def releaseVideoDevices(self):
        for dev in self.sensor_devices:
            dev.release()
        self.sensor_devices.clear()
        self.sensor_indices.clear()
