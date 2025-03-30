import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline
import io
import time

# Konfigurasi
st.set_page_config(
    page_title="Ghibli Style Free",
    layout="centered"
)

@st.cache_resource
def load_model():
    try:
        model_id = "nitrosocke/Ghibli-Diffusion"
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        # Nonaktifkan xformers dan aktifkan optimasi lain
        pipe.enable_attention_slicing()
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        return pipe
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def process_image(image, strength=0.6):
    image = image.convert("RGB")
    width, height = image.size
    ratio = 512 / max(width, height)
    new_size = (int(width * ratio), int(height * ratio))
    image = image.resize(new_size, Image.LANCZOS)
    
    pipe = load_model()
    if pipe:
        return pipe(
            prompt="ghibli style, high quality, vibrant colors",
            image=image,
            strength=strength,
            guidance_scale=7,
            num_inference_steps=25
        ).images[0]
    return None

# UI
st.title("🎨 Free Ghibli Generator")
uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns(2)
    with col1:
        img = Image.open(uploaded_file)
        st.image(img, caption="Original")
    
    with col2:
        if st.button("Transform", type="primary"):
            with st.spinner("Processing..."):
                start = time.time()
                try:
                    result = process_image(img)
                    if result:
                        st.image(result, caption="Ghibli Style")
                        buf = io.BytesIO()
                        result.save(buf, format="PNG")
                        st.download_button(
                            "Download",
                            buf.getvalue(),
                            "ghibli.png",
                            "image/png"
                        )
                        st.success(f"Done in {time.time()-start:.1f}s")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Try smaller image or refresh app")

st.warning("Note: Free tier may take 2-5 minutes per image")
