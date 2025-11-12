import os
import io
import time
import warnings
from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

# 🔑 Set Stability AI credentials
os.environ['STABILITY_KEY'] = 'sk-AITVAxnXaA28OII2yL56dedrppNy3khUVjOqdtZq8PupCSUN'

os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'

# ✅ Ask user for input
print("💍 Let's enhance your jewelry sketch with material and shine!")

image_path = input("📁 Enter the path to your reference image (e.g., 'sketch.png'): ")
thickness = input("👉 Enter desired thickness (e.g., Thin, Medium, Thick): ")
color = input("👉 Enter metal color (e.g., Rose Gold, Yellow Gold, Silver): ")
glow = input("👉 Enter glow/shine style (e.g., Subtle shine, Glossy, Matte): ")
material = input("👉 Enter metal material (e.g., 18K Rose Gold, Silver, Platinum): ")

# ✅ Build prompt without describing the design
user_prompt = (
    f"Keep this jewelry sketch same with a high-quality look. "
    f"Apply the following properties: thickness: {thickness}, color: {color}, "
    f"glow style: {glow}, material: {material}. "
    f"The design should remain as per the original sketch. "
    f"Center the jewelry, front-facing, no background, no mannequin, transparent background look."
)

print("\n🎨 Final AI Prompt:\n", user_prompt, "\n")

# ✅ Validate and load image
if not os.path.exists(image_path):
    raise FileNotFoundError(f"❌ File not found: {image_path}")

with open(image_path, "rb") as f:
    init_image = Image.open(f).convert("RGB")

# ✅ Initialize Stability API client
stability_api = client.StabilityInference(
    key=os.environ['STABILITY_KEY'],
    verbose=True,
    engine="stable-diffusion-xl-1024-v1-0",
)

# ✅ Generate enhanced image from sketch
answers = stability_api.generate(
    prompt=user_prompt,
    init_image=init_image,
    start_schedule=0.6,  # Less = more influence from original sketch
    seed=int(time.time()),
    steps=40,
    cfg_scale=7.5,  # Prompt influence; keep lower to preserve design
    width=1024,
    height=1024,
    samples=1,
    sampler=generation.SAMPLER_K_DPMPP_2M
)

# ✅ Save the output image
for resp in answers:
    for artifact in resp.artifacts:
        if artifact.finish_reason == generation.FILTER:
            warnings.warn("⚠️ Safety filter triggered. Try different values.")
        elif artifact.type == generation.ARTIFACT_IMAGE:
            img = Image.open(io.BytesIO(artifact.binary))
            filename = f"jewelry_output_{int(time.time())}.png"
            img.save(filename)
            print(f"✅ Image saved as {filename}")

