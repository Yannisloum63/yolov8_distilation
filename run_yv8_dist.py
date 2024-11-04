from ultralytics import YOLO

teacher_model = YOLO("yolov8l.pt")

student_model = YOLO("yolov8s.pt")

student_model.train(
    data="coco_3cls_mini.yaml",
    teacher=teacher_model.model, # None if you don't wanna use knowledge distillation
    distillation_loss="cwd",
    epochs=100,
    batch=2,
    workers=4,
    exist_ok=True,
)