import json

import boto3
import streamlit as st

st.title("Building with Bedrock")  # Title of the application
st.subheader("Image Generation Demo")

REGION = "us-west-2"

# # Get user input for the prompt
# prompt = st.text_input("Enter your prompt for image generation")

# List of Stable Diffusion Preset Styles
sd_presets = [
    "None",
    "3d-model",
    "analog-film",
    "anime",
    "cinematic",
    "comic-book",
    "digital-art",
    "enhance",
    "fantasy-art",
    "isometric",
    "line-art",
    "low-poly",
    "modeling-compound",
    "neon-punk",
    "origami",
    "photographic",
    "pixel-art",
    "tile-texture",
]

# Define bedrock
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=REGION,
)


# Bedrock api call to stable diffusion
def generate_image_sd(text, style):
    """
    Purpose:
        Uses Bedrock API to generate an Image
    Args/Requests:
         text: Prompt
         style: style for image
    Return:
        image: base64 string of image
    """
    body = {
        "text_prompts": [{"text": text}],
        "cfg_scale": 10,
        "seed": 0,
        "steps": 50,
        "style_preset": style,
    }

    if style == "None":
        del body["style_preset"]

    body = json.dumps(body)

    modelId = "stability.stable-diffusion-xl-v1"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("artifacts")[0].get("base64")
    return results


def generate_image_titan(text):
    """
    Purpose:
        Uses Bedrock API to generate an Image using Titan
    Args/Requests:
         text: Prompt
    Return:
        image: base64 string of image
    """
    body = {
        "textToImageParams": {"text": text},
        "taskType": "TEXT_IMAGE",
        "imageGenerationConfig": {
            "cfgScale": 10,
            "seed": 0,
            "quality": "standard",
            "width": 512,
            "height": 512,
            "numberOfImages": 1,
        },
    }

    body = json.dumps(body)

    modelId = "amazon.titan-image-generator-v1"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("images")[0]
    return results


model = st.selectbox("Select model", ["Amazon Titan", "Stable Diffusion"])

import base64
from PIL import Image
from io import BytesIO

def base64_to_image(base64_string, image_path):
    """
    Converts a base64 string to an image file.

    Args:
        base64_string (str): The base64 string representing the image data.
        image_path (str): The path where the image file should be saved.

    Returns:
        None
    """
    # Decode the base64 string
    image_data = base64.b64decode(base64_string)

    # Create a PIL Image object from the decoded data
    image = Image.open(BytesIO(image_data))

    # Save the image to the specified path
    image.save(image_path)

    print(f"saved image to {image_path}")


if __name__ == "__main__":
    # Get user input for the prompt
    prompt = st.text_input("Enter your prompt for image generation", key="user prompt")

    converted_img_path = "converted_output.jpg"

    if model == "Stable Diffusion":
        style = st.selectbox("Select style", sd_presets)
        base64_string = generate_image_sd(prompt, style)
        base64_to_image(base64_string, converted_img_path)
        st.image(converted_img_path, caption="Generated image")


