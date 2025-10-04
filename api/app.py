from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from PIL import Image
import io, os, time, warnings, base64
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
    Generate image from user prompt and return as base64 in JSON.
    """
    if not prompt:
        return JSONResponse({"error": "Prompt cannot be empty"}, status_code=400)

    try:
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

        # Return the first generated image as base64
        for resp in answers:
            for artifact in resp.artifacts:
                if artifact.finish_reason == generation.FILTER:
                    return JSONResponse({"error": "Safety filter triggered. Try a different prompt."}, status_code=400)
                elif artifact.type == generation.ARTIFACT_IMAGE:
                    img = Image.open(io.BytesIO(artifact.binary))
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                    return JSONResponse({"image": img_base64})

        return JSONResponse({"error": "No image generated"}, status_code=500)
    
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
