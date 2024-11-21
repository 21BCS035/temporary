import streamlit as st
import boto3
from datetime import datetime
import json
from decimal import Decimal
import uuid
from typing import Optional
import os
from dotenv import load_dotenv
from io import BytesIO
from image_generator import generate_image
from chatUi import ChatUI
from database import Database
# from database import Database
from dashboard import render_dashboard
from analytics import Analytics
# Load environment variables
load_dotenv()

# AWS Configuration
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

# Initialize session state
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'login'

def calculate_openai_cost(tokens_used: int) -> Decimal:
    openai_cost_per_token = Decimal('0.002')
    return Decimal(str(tokens_used)) * openai_cost_per_token

def calculate_image_generation_cost(image_count: int) -> Decimal:
    stable_diffusion_cost_per_image = Decimal('0.02')
    return Decimal(str(image_count)) * stable_diffusion_cost_per_image

def calculate_storage_cost(storage_gb: float) -> Decimal:
    s3_storage_cost_per_gb = Decimal('0.023')
    return Decimal(str(storage_gb)) * s3_storage_cost_per_gb

def generate_and_save_image(prompt: str, user_id: str, analytics) -> Optional[str]:
    try:
        s3_bucket = os.getenv('S3_BUCKET_NAME')
        # Call Stable Diffusion API
        image_data = generate_image(prompt)  # Returns a PIL.Image object
        st.image(image_data, caption="Generated Image")
        
        # Convert the PIL.Image object to bytes
        image_bytes = BytesIO()
        image_data.save(image_bytes, format='JPEG')
        image_bytes = image_bytes.getvalue()
        
        # Generate unique filename
        filename = f"{user_id}/{uuid.uuid4()}.jpeg"
        
        # Upload to S3
        s3_client.put_object(
            Bucket=s3_bucket,
            Key=filename,
            Body=image_bytes,
            ContentType='image/jpeg'
        )
        analytics.log_image_usage(user_id)
        # Log usage could be implemented here if needed
        return filename
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")
        return None

def main():
    # Initialize database
    db = Database()
    
    # Ensure tables exist
    try:
        db.create_tables()
    except Exception as e:
        st.error(f"Error creating tables: {str(e)}")

    st.sidebar.title("Navigation")
    
    if st.session_state.user_id is None:
        render_login()
        return

    page = st.sidebar.radio(
        "Select a page",
        ["Chat", "Image Generation", "Dashboard"]
    )

    if page == "Chat":
        ChatUI(db)
    elif page == "Image Generation":
        render_image_generation(db)
    elif page == "Dashboard":
        render_dashboard(db)

def render_login():
    st.title("Login")
    
    if st.button("Login"):
        # Implement actual authentication here
        st.session_state.user_id = "yogesh"
        st.rerun()

def render_image_generation(db):
    st.title("AI Image Generator")
    analytics = Analytics(db)  # Create Analytics instance

    # Prompt for new image generation
    prompt = st.text_area("Enter your image prompt")
    if st.button("Generate Image"):
        if not prompt:
            st.warning("Please enter a prompt")
            return

        with st.spinner("Generating image..."):
            image_path = generate_and_save_image(prompt, st.session_state.user_id, analytics)  # Pass analytics
            if image_path:
                st.success("Image generated successfully!")
                st.rerun()

    # Display user's generated images
    st.subheader("Your Generated Images")
    s3_bucket = os.getenv('S3_BUCKET_NAME')
    images = list_user_images(st.session_state.user_id, s3_bucket)
    
    if images:
        # Initialize session state for dialog
        if 'show_full_image' not in st.session_state:
            st.session_state.show_full_image = False
        if 'selected_image' not in st.session_state:
            st.session_state.selected_image = None
        if 'selected_image_name' not in st.session_state:
            st.session_state.selected_image_name = None

        # Create a grid layout
        cols = st.columns(4)  # Show 4 images per row
        for idx, image in enumerate(images):
            with cols[idx % 4]:
                image_data = fetch_image_from_s3(s3_bucket, image)
                if image_data:
                    # Display thumbnail
                    thumbnail = create_thumbnail(image_data, (150, 150))
                    
                    # Create a clickable container
                    if st.button("View", key=f"view_{image}"):
                        st.session_state.selected_image = image_data
                        st.session_state.selected_image_name = image.split('/')[-1]
                        st.session_state.show_full_image = True
                    
                    st.image(thumbnail, use_column_width=True)
                    
                    # Delete button below thumbnail
                    if st.button("Delete", key=f"del_{image}"):
                        delete_image_from_s3(s3_bucket, image)
                        st.success("Image deleted successfully!")
                        st.rerun()
                else:
                    st.error(f"Error fetching image: {image}")

        # Dialog for full-size image
        if st.session_state.show_full_image and st.session_state.selected_image is not None:
            with st.container():
                col1, col2 = st.columns([6, 1])
                with col1:
                    st.subheader(f"Viewing: {st.session_state.selected_image_name}")
                with col2:
                    if st.button("Close"):
                        st.session_state.show_full_image = False
                        st.session_state.selected_image = None
                        st.session_state.selected_image_name = None
                        st.rerun()
                
                st.image(st.session_state.selected_image, use_column_width=True)
    else:
        st.info("You haven't generated any images yet.")

def create_thumbnail(image_data, size):
    """Create a thumbnail of the image"""
    from PIL import Image
    from io import BytesIO

    # Convert bytes to PIL Image if needed
    if isinstance(image_data, bytes):
        image = Image.open(BytesIO(image_data))
    else:
        image = Image.open(BytesIO(image_data.getvalue()))
    
    # Create thumbnail
    image.thumbnail(size)
    
    # Convert back to bytes
    thumb_io = BytesIO()
    image.save(thumb_io, format='PNG')
    return thumb_io.getvalue()

def fetch_image_from_s3(bucket: str, key: str) -> Optional[BytesIO]:
    """Fetch image from S3 and return as BytesIO object"""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        image_data = response['Body'].read()
        return BytesIO(image_data)
    except Exception as e:
        st.error(f"Error fetching image from S3: {str(e)}")
        return None

def list_user_images(user_id: str, bucket: str) -> list:
    """Lists all the images generated by the user in the S3 bucket."""
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=f"{user_id}/")
        return [obj['Key'] for obj in response.get('Contents', [])]
    except Exception as e:
        st.error(f"Error listing user images: {str(e)}")
        return []

def delete_image_from_s3(bucket: str, key: str):
    """Deletes an image from the S3 bucket."""
    try:
        s3_client.delete_object(Bucket=bucket, Key=key)
    except Exception as e:
        st.error(f"Error deleting image from S3: {str(e)}")

if __name__ == "__main__":
    main()