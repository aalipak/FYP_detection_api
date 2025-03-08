from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from PIL import Image
import io
import torch
from pathlib import Path
import numpy as np

app = FastAPI()

# Load your YOLOv5 model (update path if needed)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./models/best.pt')
model.eval()

class PredictionResponse(BaseModel):
    boxes: list[list[float]]  # [[x1, y1, x2, y2], ...]
    scores: list[float]       # [0.95, 0.89, ...]
    labels: list[str]         # ["Acne", "Wrinkles", ...]

@app.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    results = model(image)

    predictions = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, class]

    boxes = predictions[:, :4].tolist()
    scores = predictions[:, 4].tolist()
    class_ids = predictions[:, 5].astype(int).tolist()

    labels = [model.names[i] for i in class_ids]

    return {"boxes": boxes, "scores": scores, "labels": labels}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
