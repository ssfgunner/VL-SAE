from PIL import Image
import requests
from io import BytesIO

def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return image
    except Exception as e:
        print(f"Fail to Load Images {url}: {e}")

        placeholder = Image.new('RGB', (500, 350), color='lightgray')

        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(placeholder)
        draw.text((50, 175), "Fail to Load Images", fill="red")
        return placeholder