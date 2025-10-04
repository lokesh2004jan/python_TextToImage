# api/app.py

from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image
import io, os, time
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from dotenv import load_dotenv

load_dotenv()
STABILITY_KEY = os.getenv("STABILITY_KEY")

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
    try:
        init_image = Image.open(file.file).convert("RGB")
    except Exception:
        return JSONResponse({"error": "Invalid image file"}, status_code=400)

    user_prompt = (
        f"{prompt} Apply the following properties: thickness: {thickness}, color: {color}, "
        f"glow style: {glow}, material: {material}. Center the jewelry, front-facing, no background, transparent background look."
    )

    try:
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
    except Exception as e:
        return JSONResponse({"error": f"Failed to generate image: {str(e)}"}, status_code=500)

    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.type == generation.ARTIFACT_IMAGE:
                img = Image.open(io.BytesIO(artifact.binary))
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                buffered.seek(0)
                return StreamingResponse(buffered, media_type="image/png")

    return JSONResponse({"error": "No image generated"}, status_code=500)

# Vercel will detect `app` as the ASGI entrypoint automatically
