from ultralytics import YOLO

teacher_model = YOLO("best_l_trained_on_3cls.pt")

student_model = YOLO("best_s_trained_on_3cls.pt")

student_model.train(
    data="coco_3cls_mini.yaml",
    teacher=teacher_model.model, # None if you don't wanna use knowledge distillation
    distillation_loss="cwd",
    epochs=800,
    batch=16,
    workers=8,
    exist_ok=True,
    name='13_11_test'
)
