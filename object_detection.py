import setup_path
import airsim
import cv2
import numpy as np
import time
import io
client = airsim.VehicleClient()
client.confirmConnection()
import torch as th
import torchvision.transforms as T
from PIL import Image, ImageDraw

# Assuming you have already initialized AirSim and connected to the simulator

# Initialize the model
model = th.hub.load('facebookresearch/detr', 'detr_resnet101', pretrained=True)

# Define the image transformation pipeline
transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define the class labels
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Assuming you have already initialized AirSim and connected to the simulator
# Capture images from AirSim camera and perform object detection
def detect_objects_from_airsim_camera():
    # Capture image from AirSim camera
    image_request = airsim.ImageRequest("front_center", airsim.ImageType.Scene)
    
    # Capture image from AirSim camera
    airsim_img = client.simGetImages([image_request])
    # Convert to PIL image format
    img = Image.open(io.BytesIO(airsim_img[0].image_data_uint8)).convert('RGB')
    # Resize image
    img = img.resize((800, 600))
    # Perform object detection inference
    img_tens = transform(img).unsqueeze(0)
    with th.no_grad():
        output = model(img_tens)
    # Draw predicted boxes on image
    im_with_boxes = draw_predicted_boxes(img, output)
    
    return im_with_boxes

# Draw predicted boxes on the image
def draw_predicted_boxes(img, output):
    im2 = img.copy()
    drw = ImageDraw.Draw(im2)
    pred_logits = output['pred_logits'][0][:, :len(CLASSES)]
    pred_boxes = output['pred_boxes'][0]
    max_output = pred_logits.softmax(-1).max(-1)
    topk = max_output.values.topk(15)
    pred_logits = pred_logits[topk.indices]
    pred_boxes = pred_boxes[topk.indices]
    for logits, box in zip(pred_logits, pred_boxes):
        cls = logits.argmax()
        if cls >= len(CLASSES):
            continue
        label = CLASSES[cls]
        box = box.cpu() * th.Tensor([800, 600, 800, 600])
        x, y, w, h = box
        x0, x1 = x-w//2, x+w//2
        y0, y1 = y-h//2, y+h//2
        drw.rectangle([x0, y0, x1, y1], outline='red', width=5)
        drw.text((x, y), label, fill='white')
    # Display the image with predicted boxes
    return im2

# Assuming airsim_client is initialized and connected to the simulator
while True:
    # Get image frame from AirSim
    image = detect_objects_from_airsim_camera()
    
    # Convert PIL image to OpenCV format
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Display the image with bounding boxes
    cv2.imshow("Object Detection", image_cv2)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        client.enableApiControl(False)
        break

# Clean up
cv2.destroyAllWindows()
