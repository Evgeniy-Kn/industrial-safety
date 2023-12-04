from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated
import cv2


def are_rectangles_intersecting(rect1, rect2):
    """Функция для определения пересечения двух прямоугольников"""
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    if x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2:
        cv2.putText(frame, 'Опасность!', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'Опасности нет', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


rectangle_1 = (100, 100, 100, 100)  # основная рамка (x, y, width, height)
# rectangle_2 = (150, 150, 150, 150)  # основная рамка (x, y, width, height)
# rectangle_3 = (200, 200, 200, 200)  # основная рамка (x, y, width, height)




rectangle_x = []

model = YOLO('C:/Users/knyaz/PycharmProjects/yolo_cuda/runs/detect/test_hand_v4_2_nano/weights/best.pt')  # веса
url_ip = 'http://192.168.0.102:8080/video'  # адрес камеры

# cap = cv2.VideoCapture('IMG_4298.MOV')  # ролик
# cap = cv2.VideoCapture(url_ip)  # ip камера
cap = cv2.VideoCapture(0)  # веб-камера

# для изменения разрешения
# new_width = 400
# new_height = 700

while True:
    _, frame = cap.read()

    # изменение разрешения
    # frame = cv2.resize(frame, (new_width, new_height))
    center_x = frame.shape[1] // 2
    center_y = frame.shape[0] // 2

    results = model.predict(frame)
    for r in results:

        annotator = Annotator(frame)

        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            x, y, w, h = map(int, b.cpu().numpy().tolist())  # получение координат рамок объекта
            rectangle_x.extend([x, y, w, h])
            c = box.cls
            annotator.box_label(b, model.names[int(c)])

            are_rectangles_intersecting(rectangle_1, rectangle_x)
            rectangle_x = []

    frame = annotator.result()
    frame = cv2.rectangle(frame, (rectangle_1[0], rectangle_1[1]), (rectangle_1[0] + rectangle_1[2], rectangle_1[1] + rectangle_1[3]), (0, 0, 255),
                          thickness=2, lineType=8, shift=0)

    cv2.imshow('YOLO V8 Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()