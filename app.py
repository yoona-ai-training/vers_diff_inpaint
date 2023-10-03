from potassium import Potassium, Request, Response
import cv2
from PIL import Image
from diffusers import VersatileDiffusionDualGuidedPipeline
import torch
import numpy as np
from diffusers.utils import load_image
import base64
from io import BytesIO

app = Potassium("versatile-diffusion")

@app.init
def init():
  
  pipe = VersatileDiffusionDualGuidedPipeline.from_pretrained(
      "shi-labs/versatile-diffusion"
  ).to("cuda:0")
  pipe.remove_unused_weights()

  context = {
      "pipe": pipe
  }
  
  return context

@app.handler()
def handler(context: dict, request: Request) -> Response:
    pipe = context.get("pipe")

    generator = torch.Generator(device="cuda").manual_seed(0)
    text_to_image_strength = 0.75

    image = request.json.get("image")
    prompt = request.json.get("prompt")
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
  
    
    image = pipe(prompt=prompt, image=image, text_to_image_strength=text_to_image_strength, generator=generator).images[0]
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return Response(json = {"output": img_str.decode('utf-8')}, status=200)
    
if __name__ == "__main__":
    app.serve()
