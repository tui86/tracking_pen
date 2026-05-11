import cv2
from ultralytics import YOLO
import dotenv
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


#Giảm frame để tăng tốc độ xử lý
frame_count = 0
last_drawn_frame = None

print('Bấm q để thoát màn hình')
while True:
    #Đọc từng khung hình từ webcam
    ret, frame = cap.read()
    if not ret:
        print('Không thể nhận diện khung hình, đang thoát')
        break
    
    frame_count += 1

    if frame_count % 3 == 0 or last_drawn_frame is None:

        #Đưa khung hình cho model
        results = model(frame, stream=True, verbose=False)

        for result in results:
            #Lấy thông tin khung chữ nhật
            boxes = result.boxes
            for box in boxes:
                #Lấy khung hộp
                x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
                #Lấy độ tự tin
                conf = float(box.conf[0])

                #Lấy tên vật thể
                cls = int(box.cls[0])
                class_name = model.names[cls]
                if conf > 0.65:
                    #Chụp ảnh khi phát hiện vật thể
                    if not os.path.exists('./data'):
                        os.makedirs('./data')
                    match class_name:
                        case 'ballpoint_pen':
                            count_data_pen = len(os.listdir('./data/ballpoint_pen'))
                            cv2.imwrite(f'./data/ballpoint_pen/ballpoint_pen_{count_data_pen+1}.jpg', frame)
                            print(f'Đã lưu ảnh ballpoint_pen_{count_data_pen+1}.jpg')
                        case 'pencil':
                            count_data_pencil = len(os.listdir('./data/pencil'))
                            cv2.imwrite(f'./data/pencil/pencil_{count_data_pencil+1}.jpg', frame)
                            print(f'Đã lưu ảnh pencil_{count_data_pencil+1}.jpg')

                    #Vẽ hình chữ nhật quanh vật thể
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text = f"{class_name} {conf:.2f}"
                    cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        frame_count = 0
        last_drawn_frame = frame.copy()
    cv2.imshow('Chuong trinh nhan dien', last_drawn_frame)

    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()