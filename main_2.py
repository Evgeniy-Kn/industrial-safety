from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated
import cv2
from time import time


def are_rectangles_intersecting(rect1, rect2, rect3, rectx):
    """Функция для определения пересечения прямоугольников"""
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    x3, y3, w3, h3 = rect3
    xx, yx, wx, hx = rectx

    if x1 < xx + wx and x1 + w1 > xx and y1 < yx + hx and y1 + h1 > yx:
        return 3
    elif x2 < xx + wx and x2 + w2 > xx and y2 < yx + hx and y2 + h2 > yx:
        return 2
    elif x3 < xx + wx and x3 + w3 > xx and y3 < yx + hx and y3 + h3 > yx:
        return 1
    else:
        return False


rectangle_1 = (300, 200, 100, 100)  # основная рамка (x, y, width, height)
rectangle_2 = (250, 150, 200, 200)  # основная рамка (x, y, width, height)
rectangle_3 = (200, 100, 300, 300)  # основная рамка (x, y, width, height)

rectangle_x = []  # для обнаруженных рамок при сравнении

model = YOLO('C:/Users/knyaz/PycharmProjects/yolo_cuda/runs/detect/hand_v5_medium/weights/best.pt')  # веса
# model = YOLO('yolov8s.pt')
url_ip = 'http://192.168.0.102:8080/video'  # адрес камеры

# cap = cv2.VideoCapture('IMG_4298.MOV')  # ролик
cap = cv2.VideoCapture(url_ip)  # ip камера
# cap = cv2.VideoCapture(0)  # веб-камера

# # для изменения разрешения
# new_width = 400
# new_height = 700

while True:
    start_time = time()
    _, frame = cap.read()
    count_box = [0]  # переменная для записи данных с функции по пересечению

    # изменение разрешения
    # frame = cv2.resize(frame, (new_width, new_height))

    results = model.predict(frame)
    for result in results:
        annotator = Annotator(frame)
        boxes = result.boxes
        for box in boxes:

            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            x, y, w, h = map(int, b.cpu().numpy().tolist())  # получение координат рамок объекта
            rectangle_x.extend([x, y, w, h])
            c = box.cls
            annotator.box_label(b, model.names[int(c)])

            intersection_result = are_rectangles_intersecting(rectangle_1, rectangle_2, rectangle_3, rectangle_x)
            count_box.append(intersection_result)
            print(count_box)
            rectangle_x = []

    if max(count_box) == 3:
        cv2.putText(frame, 'Опасность!', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    elif max(count_box) == 2:
        cv2.putText(frame, 'Опасность!', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    elif max(count_box) == 1:
        cv2.putText(frame, 'Опасность!', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'Опасности нет', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 55, 0), 2, cv2.LINE_AA)



    frame = annotator.result()
    frame = cv2.rectangle(frame, (rectangle_1[0], rectangle_1[1]),
                          (rectangle_1[0] + rectangle_1[2], rectangle_1[1] + rectangle_1[3]),
                          (0, 0, 255), thickness=2, lineType=8, shift=0)
    frame = cv2.rectangle(frame, (rectangle_2[0], rectangle_2[1]),
                          (rectangle_2[0] + rectangle_2[2], rectangle_2[1] + rectangle_2[3]),
                          (0, 255, 0), thickness=2, lineType=8, shift=0)
    frame = cv2.rectangle(frame, (rectangle_3[0], rectangle_3[1]),
                          (rectangle_3[0] + rectangle_3[2], rectangle_3[1] + rectangle_3[3]),
                          (255, 0, 0), thickness=2, lineType=8, shift=0)

    end_time = time()
    fps = 1 /round(end_time - start_time, 2)
    print('FPS: ' + str(fps))
    cv2.imshow('YOLO V8 Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()
