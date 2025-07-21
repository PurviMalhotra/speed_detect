import cv2
from ultralytics import YOLO, solutions

model= YOLO("yolo11n.pt")
names = model.model.names   

cap= cv2.VideoCapture("/Users/purvimalhotra/smartserv/venv/Vehicles 4K Video.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps=(int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

video_writer = cv2.VideoWriter("speed_estimation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

line_pts=[(300,1000), (2500,1000)]

speedestimator = solutions.SpeedEstimator(
    model="yolo11n.pt",
    show=True,
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break
    results = speedestimator(im0)
    video_writer.write(results.plot_im)

cap.release()
video_writer.release()
cv2.destroyAllWindows()





