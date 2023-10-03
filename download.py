from diffusers import VersatileDiffusionDualGuidedPipeline
import torch

def download_model():
  
  pipe = VersatileDiffusionDualGuidedPipeline.from_pretrained(
    "shi-labs/versatile-diffusion"
  ).to("cuda:0")
  
  pipe.remove_unused_weights()

if __name__ == "__main__":
    download_model()
