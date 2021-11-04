import sys

from gui import MainWindow
from sensor import Sensor

from PyQt5.QtWidgets import QApplication


if __name__ == "__main__":
    opencvView = False
    try:
        args = sys.argv[1:]
        if (len(args) == 1 and args[0] != "opencv_gui") or len(args) > 1:
            print(f"Invalid or too many arguments {args}.")
            print("Can be 0 arguments or 'opencv_gui'")
            sys.exit(1)
        elif len(args) == 1 and args[0] == "opencv_gui":
            opencvView = True
    except ValueError:
        print("Not enough arguments", file=sys.stderr)
        exit(-1)

    if opencvView:
        sensor = Sensor()
        try:
            sensor.startSensors()
        except Exception as e:
            print(e)
            sys.exit(1)
        sensor.openCVShow()
        print("Exiting application...")
    else:
        app = QApplication([])
        window = MainWindow()
        app.aboutToQuit.connect(window.sigint_handler)

        sys.exit(app.exec_())
        print("Exiting application...")
