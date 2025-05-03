import cv2
import numpy as np
import time
from gpiozero import CamJamKitRobot
from simple_pid import PID

robot = CamJamKitRobot()
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)