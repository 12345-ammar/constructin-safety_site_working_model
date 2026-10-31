from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
from ultralytics import YOLO
import uvicorn

app = FastAPI()

# Apna trained model load karein[cite: 8]
model = YOLO(r"C:\Users\Asim\Desktop\construction_safely_project\construction_site_safely_model.pt")

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    # Uploaded image ko read karein
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # YOLO prediction run karein
    results = model(img)
    res = results[0]

    total_instances = len(res.boxes)
    violators_count = 0

    # Classes check karein (misal ke tor par NO-Safety Vest ya missing gear)
    for box in res.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        if "NO" in cls_name or cls_name == "NO-SafeTy Vest":
            violators_count += 1

    # Annotated image (red boxes ke sath) generate karein
    annotated_img = res.plot()
    _, encoded_img = cv2.imencode('.jpg', annotated_img)
    
    return {
        "total_instances": total_instances,
        "violators_count": violators_count
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 