from ultralytics import YOLO

model = YOLO('./models/best.pt')

model.predict('./input_vidoes/08fd33_4.mp4', save = True)



