# main.py

from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
from torchvision.transforms import functional as F

import model as M


app = FastAPI()


@app.post("/images/")
async def create_upload_file(file: UploadFile = File(...)):

    contents = await file.read()  # <-- Important!

    # Convert from bytes to PIL image
    img = Image.open(BytesIO(contents))

    # Convert to tensor
    img = F.pil_to_tensor(img)

    # Predictions
    preds = M.predict(img)

    # Convert to string
    for k in preds:
        preds[k] = str(preds[k])

    return {"filename": file.filename, "predictions": preds}
