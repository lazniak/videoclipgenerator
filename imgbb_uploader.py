# imgbb_uploader.py
import requests
import base64
from PIL import Image
import io
import numpy as np

class ImgBBUploadJPG:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "jpeg_quality": ("INT", {"default": 90, "min": 1, "max": 100, "step": 1}),
                "expire": ("BOOLEAN", {"default": False}),
                "expiration_time": (
                    "INT",
                    {"default": 60, "min": 60, "max": 15552000, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("image_url",)

    FUNCTION = "upload"

    CATEGORY = "image/upload"

    def upload(self, image, api_key, jpeg_quality, expire, expiration_time):
        """
        Upload an image to ImgBB in JPG format and return the URL.
        """
        if not api_key:
            raise ValueError("API Key is required")

        # Convert image tensor to PIL Image
        img = image[0]
        img = 255.0 * img.cpu().numpy()
        img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

        # Save PIL Image to buffer as JPEG
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=jpeg_quality)
        buffer.seek(0)

        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        url = f"https://api.imgbb.com/1/upload?key={api_key}"
        if expire:
            url += f"&expiration={expiration_time}"

        payload = {
            "image": base64_image,
        }

        try:
            response = requests.post(url, data=payload)
            result = response.json()

            if result.get("success"):
                return (result["data"]["url"],)
            else:
                error_message = result.get("error", {}).get("message", "Unknown error")
                raise ValueError(f"Error: {error_message}")
        except Exception as e:
            raise ValueError(f"Error: {str(e)}")


NODE_CLASS_MAPPINGS = {"ImgBBUploadJPG": ImgBBUploadJPG}

NODE_DISPLAY_NAME_MAPPINGS = {"ImgBBUploadJPG": "Upload to ImgBB (JPG)"}