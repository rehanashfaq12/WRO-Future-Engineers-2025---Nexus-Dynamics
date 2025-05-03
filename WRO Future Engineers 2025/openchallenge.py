import cv2
import numpy as np
import time
from gpiozero import CamJamKitRobot
from simple_pid import PID

robot=CamJamKitRobot()
camera=cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH,640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
pid=PID(0.5,0.01,0.1,setpoint=320)
lap_count=0
corner_count=0
MAX_LAPS=3
prev_steering=0

def detect_lane(frame):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blurred=cv2.GaussianBlur(gray,(5,5),0)
    edges=cv2.Canny(blurred,50,150)
    lines=cv2.HoughLinesP(edges,1,np.pi/180,50,minLineLength=50,maxLineGap=50)
    left=[]; right=[]
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0]:
            slope=(y2-y1)/(x2-x1+1e-5)
            if abs(slope)<0.5: continue
            if slope<0: left.append((x1,y1,x2,y2))
            else: right.append((x1,y1,x2,y2))
    left_avg=np.mean(left,axis=0) if left else None
    right_avg=np.mean(right,axis=0) if right else None
    return left_avg, right_avg

def calculate_steering(left_line,right_line):
    if left_line is not None and right_line is not None:
        x1l,y1l,x2l,y2l=left_line
        x1r,y1r,x2r,y2r=right_line
        midl=(x1l+x2l)/2; midr=(x1r+x2r)/2
        lane_center=(midl+midr)/2
        steering=max(min(pid(lane_center)/100,1),-1)
        return steering
    if left_line: return 0.5
    if right_line: return -0.5
    return 0

def adjust_speed(steering):
    return 0.3 if abs(steering)>0.2 else 0.6

try:
    while lap_count<MAX_LAPS:
        ret,frame=camera.read()
        if not ret: break
        left_line,right_line=detect_lane(frame)
        steering=calculate_steering(left_line,right_line)
        speed=adjust_speed(steering)
        robot.left_motor.value=max(min(speed+steering,1),-1)
        robot.right_motor.value=max(min(speed-steering,1),-1)
        if abs(steering)>0.5 and abs(prev_steering)<=0.5:
            corner_count+=1
            if corner_count%4==0: lap_count+=1
        prev_steering=steering
        time.sleep(0.1)
    robot.stop()
finally:
    camera.release()
    cv2.destroyAllWindows()
