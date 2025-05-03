import cv2
import numpy as np
import time
from gpiozero import Robot, CamJamKitRobot
from simple_pid import PID

robot = CamJamKitRobot()
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

pid = PID(Kp=0.5, Ki=0.01, Kd=0.1, setpoint=320)
lap_count = 0
MAX_LAPS = 3

def detect_lane(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=50)
    
    if lines is not None:
        left_lines = []
        right_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-5)
            if abs(slope) < 0.5:
                continue
            if slope < 0:
                left_lines.append(line[0])
            else:
                right_lines.append(line[0])
        
        left_avg = np.mean(left_lines, axis=0) if left_lines else None
        right_avg = np.mean(right_lines, axis=0) if right_lines else None
        
        return left_avg, right_avg
    return None, None

def calculate_steering(left_line, right_line):
    if left_line is None or right_line is None:
        return 0
    
    x1_left, y1_left, x2_left, y2_left = left_line
    x1_right, y1_right, x2_right, y2_right = right_line
    
    mid_left = (x1_left + x2_left) / 2
    mid_right = (x1_right + x2_right) / 2
    lane_center = (mid_left + mid_right) / 2
    
    steering = pid(lane_center)
    return steering / 100

def adjust_speed(steering_angle):
    base_speed = 0.6
    if abs(steering_angle) > 20:
        return base_speed * 0.5
    return base_speed

try:
    while lap_count < MAX_LAPS:
        ret, frame = camera.read()
        if not ret:
            break
        
        left_line, right_line = detect_lane(frame)
        steering = calculate_steering(left_line, right_line)
        speed = adjust_speed(steering)
        
        robot.left_motor.value = speed + steering
        robot.right_motor.value = speed - steering
        
        time.sleep(0.1)
        if time.time() % 10 < 0.1:
            lap_count += 1

finally:
    camera.release()
    robot.stop()
    cv2.destroyAllWindows()