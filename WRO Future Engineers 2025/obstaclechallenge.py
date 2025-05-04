import cv2
import numpy as np
import time
from gpiozero import Robot, CamJamKitRobot
from simple_pid import PID

robot = CamJamKitRobot()
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

steering_pid = PID(0.5, 0.01, 0.1, setpoint=320)
speed_pid = PID(0.8, 0, 0, setpoint=0.6)

STATE_LANE_FOLLOWING = 0
STATE_PARKING = 1
current_state = STATE_LANE_FOLLOWING

RED_LOWER = np.array([0, 120, 70])
RED_UPPER = np.array([10, 255, 255])
GREEN_LOWER = np.array([35, 50, 50])
GREEN_UPPER = np.array([90, 255, 255])
MAGENTA_LOWER = np.array([140, 60, 60])
MAGENTA_UPPER = np.array([170, 255, 255])

PARKING_SAFE_DISTANCE = 50  
lap_count = 0

def detect_traffic_signs(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
   
    red_mask = cv2.inRange(hsv, RED_LOWER, RED_UPPER)
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_sign = max(red_contours, key=cv2.contourArea) if red_contours else None
    
    green_mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    green_sign = max(green_contours, key=cv2.contourArea) if green_contours else None
    
    return red_sign, green_sign

def adjust_for_sign(red_sign, green_sign, frame_width):
    if red_sign:
        x, y, w, h = cv2.boundingRect(red_sign)
        if x + w/2 < frame_width/2: 
            return 15  
    if green_sign:
        x, y, w, h = cv2.boundingRect(green_sign)
        if x + w/2 > frame_width/2:  
            return -15  
    return 0

def detect_parking_lot(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    magenta_mask = cv2.inRange(hsv, MAGENTA_LOWER, MAGENTA_UPPER)
    contours, _ = cv2.findContours(magenta_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        return cv2.minAreaRect(max(contours, key=cv2.contourArea))
    return None

def parallel_park(parking_rect):
    (x, y), (w, h), angle = parking_rect
    if w < h:  
        w, h = h, w
        angle += 90
        
    
    if w >= 200:  
        robot.left(0.3)
        time.sleep(0.5)
        robot.backward(0.4)
        time.sleep(1.2)
        robot.right(0.3)
        time.sleep(0.5)
        robot.stop()
        return True
    return False

try:
    while True:
        ret, frame = camera.read()
        if not ret:
            break
            
        if current_state == STATE_LANE_FOLLOWING:
            
            left_line, right_line = detect_lane(frame)
            steering = calculate_steering(left_line, right_line)
            
            
            red_sign, green_sign = detect_traffic_signs(frame)
            sign_offset = adjust_for_sign(red_sign, green_sign, 640)
            steering += sign_offset
            
            
            speed = speed_pid(0.6 - abs(steering/50))
            robot.left_motor.value = speed + steering/30
            robot.right_motor.value = speed - steering/30
            
            
            if passed_start_section():  
                lap_count += 1
                if lap_count >= 3:
                    current_state = STATE_PARKING
                    
        elif current_state == STATE_PARKING:
            parking_rect = detect_parking_lot(frame)
            if parking_rect and parallel_park(parking_rect):
                break  

        time.sleep(0.05)

finally:
    camera.release()
    robot.stop()
    cv2.destroyAllWindows()