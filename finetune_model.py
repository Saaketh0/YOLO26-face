# If you don't want to use Ultralytics platform use this.

import torch
from ultralytics import YOLO

model = YOLO("yolo26n.pt")  # load a pretrained model

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
batch_size = -1 # -1 = AutoBatch
workers = 8 # Num of CPU worker threads



results = model.train(data="wider_face_yolo/data.yaml", 
                      epochs=100, 
                      imgsz=640, 
                      batch = batch_size,
                      workers = workers,
                      device=device)
