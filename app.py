from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from mangum import Mangum
from PIL import Image
import io, os, time, warnings
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from dotenv import load_dotenv

# Load env vars
load_dotenv()

STABILITY_KEY = "sk-PDDADMUVmwlypOQvGJ0jQmwf61E4bEiojLBAVQPGUNHitK4I"
STABILITY_HOST = "grpc.stability.ai:443"

app = FastAPI()

stability_api = client.StabilityInference(
    key=STABILITY_KEY,
    verbose=True,
    engine="stable-diffusion-xl-1024-v1-0",
)

@app.get("/test")
def test():
    return {"status": "Server is running!"}

@app.post("/enhance-sketch")
async def enhance_sketch(
    file: UploadFile,
    prompt: str = Form(...),
    thickness: str = Form(...),
    color: str = Form(...),
    glow: str = Form(...),
    material: str = Form(...)
):
    init_image = Image.open(file.file).convert("RGB")
    user_prompt = (
        f"{prompt} Apply the following properties: thickness: {thickness}, color: {color}, "
        f"glow style: {glow}, material: {material}. Center the jewelry, front-facing, no background, transparent background look."
    )

    answers = stability_api.generate(
        prompt=user_prompt,
        init_image=init_image,
        start_schedule=0.6,
        seed=int(time.time()),
        steps=40,
        cfg_scale=7.5,
        width=1024,
        height=1024,
        samples=1,
        sampler=generation.SAMPLER_K_DPMPP_2M
    )

    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.type == generation.ARTIFACT_IMAGE:
                img = Image.open(io.BytesIO(artifact.binary))
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_str = buffered.getvalue()
                return JSONResponse({"image_bytes": list(img_str)})

    return JSONResponse({"error": "No image generated"})

# Wrap FastAPI with Mangum for serverless
handler = Mangum(app)
