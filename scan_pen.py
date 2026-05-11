import cv2
from ultralytics import YOLO
import dotenv
import time
import csv
import argparse
import os

#Tải model
model = YOLO('./weights/best.pt')

#IP webcam
IP = dotenv.get_key('.env', 'IP')
if IP is not None and IP.isdigit():
    IP = int(IP)

#Khởi tạo webcam
cap = cv2.VideoCapture(IP)
if not cap.isOpened():
    print('Không thể mở webcam')
    exit()

#Khởi tạo chế độ lưu video
weight = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./video_output/output.mp4', fourcc, 20.0, (weight, height))

#Giảm frame để tăng tốc độ xử lý
frame_count = 0
last_drawn_frame = None

#Khởi tạo bộ đếm fps
prev_time = 0

#Khởi tạo bộ đếm số lượng vật thể
count_pen = {'ballpoint_pen':set(), 'pencil':set()}
if not os.path.exists('./csv'):
    os.makedirs('./csv')
with open('./csv/inventory_log.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['ID', 'Class', 'Confidence', 'Timestamp'])

print('Bấm q để thoát màn hình')
while True:
    #Đọc từng khung hình từ webcam
    ret, frame = cap.read()
    if not ret:
        print('Không thể nhận diện khung hình, đang thoát')
        break
    
    frame_count += 1

    current_time = time.time()
    if current_time - prev_time > 0:
        fps = 1 / (current_time - prev_time)
    else:
        fps = 0
    prev_time = current_time

    if frame_count % 3 == 0 or last_drawn_frame is None:

        #Đưa khung hình cho model
        results = model.track(frame, persist=True, tracker='bytetrack.yaml', stream=True, verbose=False)

        for result in results:
            #Lấy thông tin khung chữ nhật
            boxes = result.boxes

            if boxes.id is not None:
                ids = boxes.id.cpu().numpy().astype(int)

                for box, track_id in zip(boxes, ids):
                    #Lấy khung hộp
                    x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
                    #Lấy độ tự tin
                    conf = float(box.conf[0])

                    #Lấy tên vật thể
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    if conf > 0.75:

                        if class_name in count_pen:
                            count_pen[class_name].add(track_id)

                        #Vẽ hình chữ nhật quanh vật thể
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        text = f"ID:{track_id} {class_name} {conf:.2f}"
                        cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        #Ghi nhận dữ liệu vào file csv
                        with open('./csv/inventory_log.csv', mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([track_id, class_name, conf, time.strftime("%Y-%m-%d %H:%M:%S")])

        #Hiển thị số fps trên khung hình
        fps_text = f"FPS: {int(fps)}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        #Hiển thị số lượng vât thể đã đếm
        total_ballpoint_pen = len(count_pen['ballpoint_pen'])
        total_pencil = len(count_pen['pencil'])
        count_pen_text = f"Ballpoint Pen: {total_ballpoint_pen} | Pencil: {total_pencil}"
        cv2.putText(frame, count_pen_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        frame_count = 0
        last_drawn_frame = frame.copy()
        if last_drawn_frame is not None:
            out.write(last_drawn_frame)

    cv2.imshow('Chuong trinh nhan dien', last_drawn_frame)

    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

#Ghi lại tổng số lượng vật thể đã đếm vào file csv
with open('./csv/count_pen.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Class', 'Count'])
    writer.writerow(['Ballpoint Pen', total_ballpoint_pen])
    writer.writerow(['Pencil', total_pencil])
