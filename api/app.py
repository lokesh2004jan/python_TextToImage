from fastapi import FastAPI, Form
from fastapi.responses import StreamingResponse, JSONResponse
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from PIL import Image
import io, os, time, warnings
from dotenv import load_dotenv

load_dotenv()

STABILITY_KEY = os.getenv("STABILITY_KEY") or "YOUR_KEY_HERE"

app = FastAPI()

# Initialize Stability AI client
stability_api = client.StabilityInference(
    key=STABILITY_KEY,
    verbose=True,
    engine="stable-diffusion-xl-1024-v1-0",
)

@app.get("/test")
def test():
    return {"status": "Server is running!"}

@app.post("/generate-image")
async def generate_image(prompt: str = Form(...)):
    """
    Generate image from user prompt.
    """
    if not prompt:
        return JSONResponse({"error": "Prompt cannot be empty"}, status_code=400)

    # Generate image
    answers = stability_api.generate(
        prompt=prompt,
        seed=int(time.time()),
        steps=40,
        cfg_scale=7.5,
        width=1024,
        height=1024,
        samples=1,
        sampler=generation.SAMPLER_K_DPMPP_2M
    )

    # Return the first generated image
    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                warnings.warn("⚠️ Safety filter triggered. Try a different prompt.")
            elif artifact.type == generation.ARTIFACT_IMAGE:
                img = Image.open(io.BytesIO(artifact.binary))
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                buf.seek(0)
                return StreamingResponse(buf, media_type="image/png")

    return JSONResponse({"error": "No image generated"})
