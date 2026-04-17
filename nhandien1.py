import cv2
import mediapipe as mp
import time
import numpy as np
import math
print(mp.__file__)
# Các biến lưu trạng thái
HOLD_DURATION = 2.5      
EFFECT_DURATION = 20     
STABILITY_TOLERANCE = 100 
hold_start_time = 0      
effect_start_time = 0    
freeze_bg = None         
saved_polygon = None     
previous_active_arr = None    
hands_dropped = True     
prev_frame_time = 0
# Khởi tạo
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
#Xử lý đa giác
def get_polygon_from_points(active_points, scale=1.4):
    """Tự động tạo đa giác từ các điểm chốt (Tối thiểu 3 điểm)"""
    if len(active_points) < 3:
        return None    
    pts = np.array(active_points, dtype=np.int32)
    pts = cv2.convexHull(pts).reshape(-1, 2)    #Bao màng
    center = np.mean(pts, axis=0)   #Mở rộng khung hình
    scaled_pts = center + (pts - center) * scale
    return np.int32(scaled_pts)
#Vòng lập camera
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1280, 720))
    live_frame = frame.copy() 
    cv2.namedWindow("Tracking Hand Polygon", cv2.WINDOW_NORMAL)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb)
    frame_to_show = live_frame.copy()
    # Tính FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
    prev_frame_time = new_frame_time

    # Logic ngón tay mở, khép
    open_fingers = []
    closed_fingers = []
    wrists = [] # Lưu vị trí cổ tay để dự phòng làm điểm thứ 3
    if hand_results.multi_hand_landmarks:
        h, w, c = frame.shape
        for hand_landmarks in hand_results.multi_hand_landmarks:
            wrist = hand_landmarks.landmark[0]
            wrists.append((int(wrist.x * w), int(wrist.y * h)))
            tips = [4, 8, 12, 16, 20]
            for tip_idx in tips:
                tip = hand_landmarks.landmark[tip_idx]
                joint = hand_landmarks.landmark[tip_idx - 2]
                dist_tip = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
                dist_joint = math.hypot(joint.x - wrist.x, joint.y - wrist.y)
                cx, cy = int(tip.x * w), int(tip.y * h)
                # Phân loại Mở/Khép
                if dist_tip > dist_joint:
                    open_fingers.append((cx, cy))
                else:
                    closed_fingers.append((cx, cy))
    else:
        hands_dropped = True
        hold_start_time = 0
        previous_active_arr = None

    # Màu sắc 
    active_points = []
    num_open = len(open_fingers)
    if num_open == 1:       #Không tạo hình nếu chỉ có 1 ngón tay
        for pt in open_fingers + closed_fingers:
            cv2.circle(frame_to_show, pt, 6, (0, 0, 255), -1, lineType=cv2.LINE_AA)
    elif num_open >= 2:     #Tạo hình khi có trên 2 ngón tay trở lên
        for pt in open_fingers:     #Cho 1 chấm với ngón tay mở
            cv2.circle(frame_to_show, pt, 6, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        for pt in closed_fingers:   #Cho 1 chấm với ngón tay đóng
            cv2.circle(frame_to_show, pt, 6, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        active_points = open_fingers.copy()
    if freeze_bg is not None and len(active_points) >= 3 and hands_dropped:
        freeze_bg = None
        saved_polygon = None
        hold_start_time = 0
        previous_active_arr = None
        hands_dropped = False 

    # Chế độ ngắm chụp
    if freeze_bg is None:
        if len(active_points) >= 3:
            current_active_arr = np.array(active_points, dtype=np.int32)
            is_stable = False
            
            if previous_active_arr is not None and previous_active_arr.shape == current_active_arr.shape:
                diff = np.max(np.abs(current_active_arr - previous_active_arr))
                if diff < STABILITY_TOLERANCE:
                    is_stable = True
            previous_active_arr = current_active_arr
            current_polygon = get_polygon_from_points(active_points)
            if current_polygon is not None:
                cv2.polylines(frame_to_show, [current_polygon], isClosed=True, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
                min_x_poly = np.min(current_polygon[:, 0])
                max_x_poly = np.max(current_polygon[:, 0])
                min_y_poly = np.min(current_polygon[:, 1])
                if is_stable:
                    if hold_start_time == 0: hold_start_time = time.time() 
                    held_for = time.time() - hold_start_time
                    progress = min(1.0, held_for / HOLD_DURATION)                   
                    bar_width = int((max_x_poly - min_x_poly) * progress)
                    cv2.rectangle(frame_to_show, (min_x_poly, min_y_poly - 10), (min_x_poly + bar_width, min_y_poly - 5), (0, 255, 0), -1, lineType=cv2.LINE_AA)
                    cv2.putText(frame_to_show, "Dang giu yen...", (min_x_poly, min_y_poly - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, lineType=cv2.LINE_AA)
                    if held_for >= HOLD_DURATION:
                        freeze_bg = live_frame.copy()
                        saved_polygon = current_polygon
                        effect_start_time = time.time()
                        hold_start_time = 0 
                        previous_active_arr = None
                        hands_dropped = False 
                else:
                    hold_start_time = 0
        else:
            hold_start_time = 0
            previous_active_arr = None

    # Xử lý hiệu ứng và vẽ hình tròn khi tạo thành công
    if freeze_bg is not None and saved_polygon is not None:
        time_left = EFFECT_DURATION - (time.time() - effect_start_time)
        if time_left > 0:
            mask = np.zeros(frame_to_show.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [saved_polygon], 255, lineType=cv2.LINE_AA)
            mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            frame_to_show = np.where(mask_3d == 255, live_frame, freeze_bg)
            # Đa giác hiệu ứng 
            cv2.polylines(frame_to_show, [saved_polygon], isClosed=True, color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA)
            cv2.putText(frame_to_show, f"Hieu ung: {int(time_left)}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, lineType=cv2.LINE_AA)
            # Vòng tròn Tím bao sát khung 
            (cx_circle, cy_circle), radius = cv2.minEnclosingCircle(np.float32(saved_polygon))
            cv2.circle(frame_to_show, (int(cx_circle), int(cy_circle)), int(radius), (255, 0, 255), 1, lineType=cv2.LINE_AA)   
        else:
            freeze_bg = None
            saved_polygon = None
            hands_dropped = True
    # Hiển thị FPS
    cv2.putText(frame_to_show, f"FPS: {int(fps)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, lineType=cv2.LINE_AA)
    cv2.imshow("Tracking Hand Polygon", frame_to_show)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()