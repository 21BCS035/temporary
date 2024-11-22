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

def calculate_costs(tokens_used: int, image_count: int, storage_gb: float) -> dict:
    """Calculate various usage costs"""
    return {
        'chat': calculate_openai_cost(tokens_used),
        'images': calculate_image_generation_cost(image_count),
        'storage': calculate_storage_cost(storage_gb)
    }

def calculate_openai_cost(tokens_used: int) -> Decimal:
    return Decimal(str(tokens_used)) * Decimal('0.002')

def calculate_image_generation_cost(image_count: int) -> Decimal:
    return Decimal(str(image_count)) * Decimal('0.02')

def calculate_storage_cost(storage_gb: float) -> Decimal:
    return Decimal(str(storage_gb)) * Decimal('0.023')

def generate_and_save_image(prompt: str, user_id: str, analytics) -> Optional[str]:
    try:
        s3_bucket = os.getenv('S3_BUCKET_NAME')
        with st.spinner("🎨 Creating your masterpiece..."):
            image_data = generate_image(prompt)
            
            # Create columns for better layout
            col1, col2 = st.columns([3, 1])
            with col1:
                st.image(image_data, caption="Generated Image", use_column_width=True)
            with col2:
                st.success("✨ Image generated successfully!")
        
        # Save image to S3
        image_bytes = BytesIO()
        image_data.save(image_bytes, format='JPEG')
        image_bytes = image_bytes.getvalue()
        filename = f"{user_id}/{uuid.uuid4()}.jpeg"
        
        s3_client.put_object(
            Bucket=s3_bucket,
            Key=filename,
            Body=image_bytes,
            ContentType='image/jpeg'
        )
        analytics.log_image_usage(user_id)
        return filename
    except Exception as e:
        st.error(f"🚫 Error generating image: {str(e)}")
        return None

def render_login():
    """Render an enhanced login page"""
    st.markdown("""
        <style>
        .login-container {
            max-width: 400px;
            margin: 0 auto;
            padding: 2rem;
            background-color: #61676b;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .login-header {
            text-align: center;
            color: #1E88E5;
            margin-bottom: 2rem;
        }
        .login-button {
            width: 100%;
            background-color: #1E88E5;
            color: white;
            padding: 0.5rem;
            border-radius: 5px;
            border: none;
            cursor: pointer;
        }
        </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown("""
            <div class="login-container">
                <div class="login-header">
                    <h1>👋 Welcome to Messenger Prime</h1>
                    <p></p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("Welcome", key="login_button", help="Click to login"):
            st.session_state.user_id = "yogesh"
            st.rerun()

def render_image_generation(db):
    """Render enhanced image generation page"""
    st.markdown("""
        <style>
        .image-gen-container {
            background-color: #61676b;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .image-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 1rem;
            padding: 1rem;
        }
        .image-card {
            background-color: white;
            border-radius: 10px;
            padding: 0.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🎨 AI Image Generator")
    analytics = Analytics(db)

    with st.container():
        st.markdown("""
            <div class="image-gen-container">
                <h3>Create New Image</h3>
            </div>
        """, unsafe_allow_html=True)
        
        prompt = st.text_area("Describe your image", placeholder="Enter a detailed description of the image you want to create...")
        col1, col2, col3 = st.columns([3, 1, 1])
        with col2:
            if st.button("🎨 Generate", help="Generate a new image"):
                if not prompt:
                    st.warning("⚠️ Please enter a prompt")
                    return
                image_path = generate_and_save_image(prompt, st.session_state.user_id, analytics)
                if image_path:
                    st.rerun()

    # Display generated images
    st.subheader("🖼️ Your Gallery")
    s3_bucket = os.getenv('S3_BUCKET_NAME')
    images = list_user_images(st.session_state.user_id, s3_bucket)
    
    if images:
        initialize_session_state()
        
        # Create image grid
        cols = st.columns(4)
        for idx, image in enumerate(images):
            with cols[idx % 4]:
                display_image_card(image, s3_bucket)

        # Full-size image dialog
        display_full_size_dialog(s3_bucket)
    else:
        st.info("✨ Generate your first image to start building your gallery!")

def initialize_session_state():
    """Initialize session state variables for image viewing"""
    if 'show_full_image' not in st.session_state:
        st.session_state.show_full_image = False
    if 'selected_image' not in st.session_state:
        st.session_state.selected_image = None
    if 'selected_image_name' not in st.session_state:
        st.session_state.selected_image_name = None

def display_image_card(image, s3_bucket):
    """Display an individual image card"""
    image_data = fetch_image_from_s3(s3_bucket, image)
    if image_data:
        thumbnail = create_thumbnail(image_data, (150, 150))
        
        st.markdown("""
            <div class="image-card">
        """, unsafe_allow_html=True)
        
        st.image(thumbnail, use_column_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👁️ View", key=f"view_{image}"):
                st.session_state.selected_image = image_data
                st.session_state.selected_image_name = image.split('/')[-1]
                st.session_state.show_full_image = True
        with col2:
            if st.button("🗑️ Delete", key=f"del_{image}"):
                delete_image_from_s3(s3_bucket, image)
                st.success("🗑️ Image deleted!")
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)

def display_full_size_dialog(s3_bucket):
    """Display dialog for full-size image viewing"""
    if st.session_state.show_full_image and st.session_state.selected_image:
        with st.container():
            col1, col2 = st.columns([6, 1])
            with col1:
                st.subheader(f"📸 {st.session_state.selected_image_name}")
            with col2:
                if st.button("✖️ Close"):
                    st.session_state.show_full_image = False
                    st.session_state.selected_image = None
                    st.session_state.selected_image_name = None
                    st.rerun()
            
            st.image(st.session_state.selected_image, use_column_width=True)

def main():
    """Main application function"""
    st.set_page_config(
        page_title="Messenger Prime",
        page_icon="💭",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Add global styles
    st.markdown("""
        <style>
        .stApp {
            background-color: #2a2b2b;
        }
        .main-nav {
            background-color: white;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)

    db = Database()
    try:
        db.create_tables()
    except Exception as e:
        st.error(f"Database Error: {str(e)}")

    if st.session_state.user_id is None:
        render_login()
        return

    # Sidebar navigation
    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; padding: 1rem;'>
                <h2>🚀 Navigation</h2>
            </div>
        """, unsafe_allow_html=True)
        
        page = st.radio(
            "",
            ["💬 Chat", "🎨 Image Generation", "📊 Dashboard"],
            key="nav_radio"
        )

    # Main content
    if "💬 Chat" in page:
        ChatUI(db)
    elif "🎨 Image Generation" in page:
        render_image_generation(db)
    elif "📊 Dashboard" in page:
        render_dashboard(db)

# Helper functions remain unchanged
def create_thumbnail(image_data, size):
    from PIL import Image
    from io import BytesIO

    if isinstance(image_data, bytes):
        image = Image.open(BytesIO(image_data))
    else:
        image = Image.open(BytesIO(image_data.getvalue()))
    
    image.thumbnail(size)
    thumb_io = BytesIO()
    image.save(thumb_io, format='PNG')
    return thumb_io.getvalue()

def fetch_image_from_s3(bucket: str, key: str) -> Optional[BytesIO]:
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        return BytesIO(response['Body'].read())
    except Exception as e:
        st.error(f"S3 Error: {str(e)}")
        return None

def list_user_images(user_id: str, bucket: str) -> list:
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=f"{user_id}/")
        return [obj['Key'] for obj in response.get('Contents', [])]
    except Exception as e:
        st.error(f"S3 Error: {str(e)}")
        return []

def delete_image_from_s3(bucket: str, key: str):
    try:
        s3_client.delete_object(Bucket=bucket, Key=key)
    except Exception as e:
        st.error(f"S3 Error: {str(e)}")

if __name__ == "__main__":
    main()