import cv2
import mediapipe as mp
import time
import numpy as np

print(mp.__file__)
mp_drawing = mp.solutions.drawing_utils

# ====== CÁC BIẾN LƯU TRẠNG THÁI ======
HOLD_DURATION = 2.5      
EFFECT_DURATION = 20     
STABILITY_TOLERANCE = 60 # Dung sai rung tay (pixel)

hold_start_time = 0      
effect_start_time = 0    
freeze_bg = None         
saved_polygon = None     # Lưu toạ độ đa giác tuỳ biến thay vì ô vuông  
previous_polygon = None      
hands_dropped = True     

# ====== KHỞI TẠO ======
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
pose = mp_pose.Pose()

# --- HÀM HỖ TRỢ XỬ LÝ HÌNH DÁNG THEO TAY ---
def order_points(pts):
    """Sắp xếp 4 điểm theo chiều kim đồng hồ để nối viền không bị chéo nhau (hình nơ)"""
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:,1] - center[1], pts[:,0] - center[0])
    sorted_indices = np.argsort(angles)
    return pts[sorted_indices]

def scale_polygon(pts, scale=1.4):
    """Nới rộng khung hình từ tâm ra một chút để có chỗ trống hiển thị (padding)"""
    center = np.mean(pts, axis=0)
    scaled_pts = center + (pts - center) * scale
    return np.int32(scaled_pts)


# ====== VÒNG LẶP CAMERA ======
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (640, 480))
    live_frame = frame.copy() 
    
    cv2.namedWindow("Tracking Face - Hand - Body", cv2.WINDOW_NORMAL)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    hand_results = hands.process(rgb)
    frame_to_show = live_frame.copy()

    current_hands_count = len(hand_results.multi_hand_landmarks) if hand_results.multi_hand_landmarks else 0

    if current_hands_count < 2:
        hands_dropped = True
        hold_start_time = 0
        previous_polygon = None

    # ====================================================================
    # 1. KHI CÓ 2 TAY XUẤT HIỆN
    # ====================================================================
    if current_hands_count == 2:
        h, w, c = frame.shape
        tay1 = hand_results.multi_hand_landmarks[0]
        tay2 = hand_results.multi_hand_landmarks[1]

        # Lấy tọa độ 2 Ngón Trỏ và 2 Ngón Cái
        x_tro1, y_tro1 = int(tay1.landmark[8].x * w), int(tay1.landmark[8].y * h)
        x_cai1, y_cai1 = int(tay1.landmark[4].x * w), int(tay1.landmark[4].y * h)
        x_tro2, y_tro2 = int(tay2.landmark[8].x * w), int(tay2.landmark[8].y * h)
        x_cai2, y_cai2 = int(tay2.landmark[4].x * w), int(tay2.landmark[4].y * h)

        # Tập hợp thành 4 đỉnh của một hình đa giác
        raw_pts = np.array([
            [x_tro1, y_tro1], [x_cai1, y_cai1], 
            [x_tro2, y_tro2], [x_cai2, y_cai2]
        ], dtype=np.int32)
        
        # Sắp xếp lại để tạo hình khối hoàn chỉnh và nới rộng ra một chút
        ordered_pts = order_points(raw_pts)
        current_polygon = scale_polygon(ordered_pts, scale=1.4)

        if freeze_bg is not None and hands_dropped:
            freeze_bg = None
            saved_polygon = None
            hold_start_time = 0
            previous_polygon = None
            hands_dropped = False 

        # --- CHẾ ĐỘ NGẮM CHỤP ---
        if freeze_bg is None:
            is_stable = False
            if previous_polygon is not None:
                # So sánh độ xê dịch của cả 4 đỉnh
                diff = np.max(np.abs(current_polygon - previous_polygon))
                if diff < STABILITY_TOLERANCE:
                    is_stable = True

            previous_polygon = current_polygon

            # VẼ ĐA GIÁC NGẮM (MÀU XANH LÁ)
            cv2.polylines(frame_to_show, [current_polygon], isClosed=True, color=(0, 255, 0), thickness=2)

            # Lấy vị trí để vẽ thanh loading bar (Nằm trên đỉnh cao nhất của đa giác)
            min_x_poly = np.min(current_polygon[:, 0])
            max_x_poly = np.max(current_polygon[:, 0])
            min_y_poly = np.min(current_polygon[:, 1])

            if is_stable:
                if hold_start_time == 0:
                    hold_start_time = time.time() 
                
                held_for = time.time() - hold_start_time
                progress = min(1.0, held_for / HOLD_DURATION)
                
                # Hiển thị thanh loading chạy dần
                bar_width = int((max_x_poly - min_x_poly) * progress)
                cv2.rectangle(frame_to_show, (min_x_poly, min_y_poly - 15), (min_x_poly + bar_width, min_y_poly - 5), (0, 255, 0), -1)
                if is_stable:
                    if hold_start_time == 0:
                        hold_start_time = time.time() 
                
                held_for = time.time() - hold_start_time
                progress = min(1.0, held_for / HOLD_DURATION)
                
                # Hiển thị thanh loading chạy dần
                bar_width = int((max_x_poly - min_x_poly) * progress)
                cv2.rectangle(frame_to_show, (min_x_poly, min_y_poly - 15), (min_x_poly + bar_width, min_y_poly - 5), (0, 255, 0), -1)

                # ==========================================
                # THÊM DÒNG CHỮ CỦA BẠN VÀO NGAY DƯỚI ĐÂY:
                # ==========================================
                cv2.putText(frame_to_show, "Dang giu yen...", (min_x_poly, min_y_poly - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # CHỤP NẾU GIỮ ĐỦ 2.5 GIÂY
                if held_for >= HOLD_DURATION:
                    freeze_bg = live_frame.copy()
                    saved_polygon = current_polygon
                    effect_start_time = time.time()
                    hold_start_time = 0 
                    previous_polygon = None
                    hands_dropped = False 
            else:
                hold_start_time = 0

        # Vẽ xương 2 tay
        mp_drawing.draw_landmarks(frame_to_show, tay1, mp_hands.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame_to_show, tay2, mp_hands.HAND_CONNECTIONS)

    elif current_hands_count == 1 and freeze_bg is None:
        mp_drawing.draw_landmarks(frame_to_show, hand_results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

    # ====================================================================
    # 2. XỬ LÝ DUY TRÌ HIỆU ỨNG (CẮT THEO ĐA GIÁC MẶT NẠ)
    # ====================================================================
    if freeze_bg is not None and saved_polygon is not None:
        time_left = EFFECT_DURATION - (time.time() - effect_start_time)
        if time_left > 0:
            # Tạo một mặt nạ (mask) đen hoàn toàn
            mask = np.zeros(frame_to_show.shape[:2], dtype=np.uint8)
            
            # Tô màu trắng (255) cho vùng đa giác đã chụp
            cv2.fillPoly(mask, [saved_polygon], 255)
            
            # Chuyển mask sang 3 kênh màu để tương thích với ảnh camera
            mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            
            # Kỹ thuật xịn: Ghép phần live_frame (vào chỗ trắng) và freeze_bg (vào chỗ đen)
            frame_to_show = np.where(mask_3d == 255, live_frame, freeze_bg)
            
            # Vẽ viền màu Vàng cho đa giác
            cv2.polylines(frame_to_show, [saved_polygon], isClosed=True, color=(0, 255, 255), thickness=3)
            cv2.putText(frame_to_show, f"Hieu ung: {int(time_left)}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            freeze_bg = None
            saved_polygon = None

    # ====== FACE & POSE DETECTION ======
    gray = cv2.cvtColor(live_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w_face, h_face) in faces:
        cv2.rectangle(frame_to_show, (x, y), (x+w_face, y+h_face), (255, 0, 0), 2) 

    pose_results = pose.process(rgb)
    if pose_results.pose_landmarks:
        h_shape, w_shape, c_shape = frame_to_show.shape
        xs = [lm.x for lm in pose_results.pose_landmarks.landmark]
        ys = [lm.y for lm in pose_results.pose_landmarks.landmark]

        x_min_pose = int(min(xs) * w_shape)
        y_min_pose = int(min(ys) * h_shape)
        x_max_pose = int(max(xs) * w_shape)
        y_max_pose = int(max(ys) * h_shape)

        cv2.rectangle(frame_to_show, (x_min_pose, y_min_pose), (x_max_pose, y_max_pose), (255, 0, 255), 2) 

    # ====== HIỂN THỊ ======
    cv2.imshow("Tracking Face - Hand - Body", frame_to_show)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ====== GIẢI PHÓNG ======
cap.release()
cv2.destroyAllWindows()