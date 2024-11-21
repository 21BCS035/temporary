import requests
import io
import streamlit as st
from PIL import Image
from analytics import Analytics

API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content


# You can access the image with PIL.Image for example

def generate_image(prompt):
    image_bytes = query({
	"inputs": f"{prompt}",
	})
    user_id = st.session_state.user_id
    # analytics = Analytics(db)
    # analytics.log_image_usage(user_id)
    
    # # Log storage usage (assuming 1MB per image)
    # analytics.log_storage_usage(user_id, 0.001) 
    image = Image.open(io.BytesIO(image_bytes))
    return image
