# api/app.py

from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
from mangum import Mangum
from PIL import Image
import io, os, time
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
STABILITY_KEY = os.getenv("STABILITY_KEY")

# Initialize FastAPI
app = FastAPI()

# Initialize Stability AI client
stability_api = client.StabilityInference(
    key=STABILITY_KEY,
    verbose=True,
    engine="stable-diffusion-xl-1024-v1-0",
)

# Health check endpoint
@app.get("/test")
def test():
    return {"status": "Server is running!"}

# Sketch enhancement endpoint
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
        # Open uploaded image
        init_image = Image.open(file.file).convert("RGB")
    except Exception:
        return JSONResponse({"error": "Invalid image file"}, status_code=400)

    # Construct prompt for Stability AI
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

    # Convert generated image to PNG and return as streaming response
    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.type == generation.ARTIFACT_IMAGE:
                img = Image.open(io.BytesIO(artifact.binary))
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                buffered.seek(0)
                return StreamingResponse(buffered, media_type="image/png")

    return JSONResponse({"error": "No image generated"}, status_code=500)

# Wrap FastAPI with Mangum for serverless deployment
handler = Mangum(app)
