from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated
import cv2
from time import time


# Функция для проверки пересечения кругов и прямоугольника
def check_intersection(circle_centers, circle_radius, rectangle_x):
    rect_x, rect_y, rect_w, rect_h = rectangle_x
    for i, (center, radius) in enumerate(zip(circle_centers, circle_radius)):
        circle_distance_x = abs(center[0] - (rect_x + rect_w / 2))
        circle_distance_y = abs(center[1] - (rect_y + rect_h / 2))

        if circle_distance_x > (rect_w / 2 + radius) or circle_distance_y > (rect_h / 2 + radius):
            continue

        if circle_distance_x <= (rect_w / 2) or circle_distance_y <= (rect_h / 2):
            return i + 1

        corner_distance_sq = (circle_distance_x - rect_w / 2) ** 2 + (circle_distance_y - rect_h / 2) ** 2
        if corner_distance_sq <= (radius ** 2):
            return i + 1

    return -1


circle_x_0, circle_y_0, circle_r_0 = 250, 250, 100
circle_x_1, circle_y_1, circle_r_1 = 250, 250, 200
circle_x_2, circle_y_2, circle_r_2 = 250, 250, 300

circle_centers = [(circle_x_0, circle_y_0), (circle_x_1, circle_y_1), (circle_x_2, circle_y_2)]
circle_radius = [circle_r_0, circle_r_1, circle_r_2]

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
intersection_result = 0
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

            intersection_result = check_intersection(circle_centers, circle_radius, rectangle_x)
            count_box.append(intersection_result)
            print(count_box)
            rectangle_x = []

    if max(count_box) == 1:
        cv2.putText(frame, 'Опасность!', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    elif max(count_box) == 2:
        cv2.putText(frame, 'Опасность!', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    elif max(count_box) == 3:
        cv2.putText(frame, 'Опасность!', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'Опасности нет', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 55, 0), 2, cv2.LINE_AA)

    intersection_result = 0

    frame = annotator.result()
    cv2.circle(frame, (circle_x_0, circle_y_0), circle_r_0, (0, 0, 255), 2)
    cv2.circle(frame, (circle_x_1, circle_y_1), circle_r_1, (0, 255, 0), 2)
    cv2.circle(frame, (circle_x_2, circle_y_2), circle_r_2, (255, 0, 0), 2)

    end_time = time()
    fps = 1 /round(end_time - start_time, 2)
    print('FPS: ' + str(fps))
    cv2.imshow('YOLO V8 Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()
