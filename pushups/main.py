import cv2
import numpy as np
import time
from ultralytics import YOLO

def get_angle(a, b, c):
    cb = np.atan2(c[1] - b[1], c[0] - b[0])
    ab = np.atan2(a[1] - b[1], a[0] - b[0])
    angle = np.rad2deg(cb-ab)
    angle = angle + 360 if angle < 0 else angle
    return 360 - angle if angle > 180 else angle

def detect_pushup_phase(keypoints): #человек ща сверху или снизу
    if not keypoints or len(keypoints) == 0:
        return None
    
    kp = keypoints[0]  

    left_shoulder = kp[5]
    right_shoulder = kp[6]
    left_elbow = kp[7]
    right_elbow = kp[8]
    left_wrist = kp[9]
    right_wrist = kp[10]

    if (kp[5][2] < 0.5 or kp[6][2] < 0.5 or 
        kp[7][2] < 0.5 or kp[8][2] < 0.5): #локти и плечи смотрим уверенность в них
        return None

    left_angle = get_angle(left_shoulder, left_elbow, left_wrist)
    right_angle = get_angle(right_shoulder, right_elbow, right_wrist)

    if left_angle < 100 or right_angle < 100:
        return 'down'
    elif left_angle > 150 or right_angle > 150:
        return 'up'
    
    return None


model = YOLO("yolo26n-pose.pt") 
camera = cv2.VideoCapture(1)

pushup_count = 0
was_down = False  
last_person_time = time.time() 
person_timeout = 3.0  # три(или лучше 5) секунды до сброса счетчика если в кадре нет человека

print("клавиша q = выход, r - сброс счета ")

while camera.isOpened():
    ret, frame = camera.read()
    current_time = time.time()

    results = model(frame, verbose=False)
    
    person_detected = False

    if results and len(results) > 0:
        result = results[0]
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            keypoints_data = result.keypoints.data
            keypoints_conf = result.keypoints.conf
            
            if len(keypoints_data) > 0:
                person_detected = True
                last_person_time = current_time
                keypoints_list = []
                for i in range(len(keypoints_data[0])):
                    x = float(keypoints_data[0][i][0])
                    y = float(keypoints_data[0][i][1])
                    conf = float(keypoints_conf[0][i]) if keypoints_conf is not None else 1.0
                    keypoints_list.append([x, y, conf])
                
                phase = detect_pushup_phase([keypoints_list])
                
                # подсчет
                if phase == 'down' and not was_down:
                    was_down = True
                elif phase == 'up' and was_down:
                    pushup_count += 1
                    was_down = False
                    print(f"Всего: {pushup_count}")
                # палки скелета
                annotated = result.plot()
                frame = annotated
                # снизу или сверху сейчас человек
                if phase == 'down':
                    cv2.putText(frame, "PHASE: DOWN - PUSH UP!", (10, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                elif phase == 'up':
                    cv2.putText(frame, "PHASE: UP - GO DOWN", (10, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    if not person_detected and (current_time - last_person_time) > person_timeout:
        if pushup_count > 0:
            print(f"Человек не обнаружен {person_timeout} секунд. Сброс счетчика.")
            pushup_count = 0
            was_down = False
        last_person_time = current_time
    
    # счетчик
    cv2.putText(frame, f"PUSHUPS: {pushup_count}", (10, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    # время до сброса
    if not person_detected and pushup_count > 0:
        time_left = max(0, person_timeout - (current_time - last_person_time))
        cv2.putText(frame, f"Reset in: {time_left:.1f}s", (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    cv2.imshow("Push-up Counter", frame)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        pushup_count = 0
        was_down = False
        print("Счетчик сброшен")

camera.release()
cv2.destroyAllWindows()
print(f"Всего отжиманий: {pushup_count}")