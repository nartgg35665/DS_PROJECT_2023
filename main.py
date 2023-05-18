from flask import Flask, request, jsonify
from PIL import Image
import io
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import easyocr

app = Flask(__name__)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_ft = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights = None)
model_ft.roi_heads.box_predictor = FastRCNNPredictor(1024, 2) #1

net = model_ft.to(device)
net.load_state_dict(torch.load('weight2.pth'))

reader = easyocr.Reader(['th'])

@app.route('/process-image', methods=['POST'])
def process_image():
    # Check if an image file is present in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'})

    image_file = request.files['image']

    # Read the image file
    image = Image.open(image_file)

    # Perform image processing tasks
    # ...

    # Return the processed image as a response
    processed_image = # Your image processing code

    # Convert the processed image to bytes
    byte_stream = io.BytesIO()
    processed_image.save(byte_stream, format='PNG')
    byte_stream.seek(0)

    return send_file(byte_stream, mimetype='image/png')

if __name__ == '__main__':
    app.run()
