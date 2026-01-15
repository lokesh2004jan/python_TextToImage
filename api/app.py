from fastapi import FastAPI, Form
from fastapi.responses import StreamingResponse, JSONResponse
import requests
import os
import io
import time

app = FastAPI()

STABILITY_KEY = os.getenv("STABILITY_KEY")
@app.get("/")
def health():
    return {"status": "API is running"}

@app.post("/generate-image")
async def generate_image(prompt: str = Form(...)):
    if not prompt:
        return JSONResponse(
            {"error": "Prompt cannot be empty"},
            status_code=400
        )

    url = "https://api.stability.ai/v2beta/stable-image/generate/sd3"

    headers = {
        "Authorization": f"Bearer {STABILITY_KEY}",
        "Accept": "image/png"
    }

    files = {
        "prompt": (None, prompt),
        "output_format": (None, "png"),
        "seed": (None, str(int(time.time()))),
        "cfg_scale": (None, "7.5"),
        "steps": (None, "40")
    }

    response = requests.post(
        url,
        headers=headers,
        files=files,
        timeout=60
    )

    if response.status_code != 200:
        return JSONResponse(
            {"error": response.text},
            status_code=response.status_code
        )

    return StreamingResponse(
        io.BytesIO(response.content),
        media_type="image/png"
    )
