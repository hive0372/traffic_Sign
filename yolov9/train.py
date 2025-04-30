import torch
import cv2
import numpy as np

# Hypothetical YOLOv9 inference class.
class YOLOv9:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        # Load the pre-trained model weights (ensure you have a matching architecture defined)
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()  # Set the model to evaluation mode

    def preprocess(self, image):
        # Resize and convert the image to RGB
        img = cv2.resize(image, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Change shape from HWC to CHW and normalize pixel values
        img = img.transpose((2, 0, 1)) / 255.0
        # Add batch dimension and convert to tensor
        img_tensor = torch.from_numpy(np.expand_dims(img, axis=0)).float().to(self.device)
        return img_tensor

    def postprocess(self, outputs, conf_threshold=0.5):
        # For demonstration, assume outputs is a tensor with rows:
        # [x1, y1, x2, y2, confidence, class_id]
        outputs = outputs.cpu().detach().numpy()
        detections = []
        for output in outputs:
            x1, y1, x2, y2, conf, class_id = output
            if conf >= conf_threshold:
                detections.append([int(x1), int(y1), int(x2), int(y2), conf, int(class_id)])
        return detections

    def detect(self, image):
        # Preprocess the input image
        img_tensor = self.preprocess(image)
        # Perform inference (adjust indexing as required by your model's output format)
        with torch.no_grad():
            outputs = self.model(img_tensor)[0]
        # Postprocess the outputs to filter by confidence threshold
        detections = self.postprocess(outputs)
        return detections

def main():
    # Path to the YOLOv9 weights (ensure you have a matching model architecture)
    model_path = 'yolov9_weights.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize the YOLOv9 model
    yolo = YOLOv9(model_path, device=device)

    # Load an image for inference
    image_path = 'test.jpg'
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found!")
        return

    # Get detections
    detections = yolo.detect(image)

    # Draw detections on the image
    for (x1, y1, x2, y2, conf, class_id) in detections:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'ID:{class_id} {conf:.2f}'
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("YOLOv9 Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from ultralytics import YOLO

# loading a pre-trained model
# if the first time loading a model, it will first download the model in the directory
# available pre-trained models are YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x
model = YOLO("yolov9m.pt")

# will throw an exception if false
model._check_is_pytorch_model()

data_yaml_path = "data.yaml"

# Use 'cpu' for device since you don't have CUDA available
model.train(data=data_yaml_path,
            epochs=150,
            imgsz=100,
            device='cpu')