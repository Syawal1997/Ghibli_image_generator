import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline
import io
import time

# Konfigurasi minimal
st.set_page_config(
    page_title="Ghibli Magic Free",
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
        pipe = pipe.to("cpu")  # Force CPU only
        pipe.enable_attention_slicing()
        return pipe
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

def process_image(image):
    image = image.convert("RGB").resize((512, 512))
    pipe = load_model()
    if pipe:
        return pipe(
            prompt="ghibli style, high quality",
            image=image,
            strength=0.6,
            guidance_scale=7,
            num_inference_steps=20  # Lebih sedikit untuk free tier
        ).images[0]
    return None

# UI Minimalis
st.title("Free Ghibli Converter")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Original Image")
    
    if st.button("Convert to Ghibli"):
        with st.spinner("Processing (may take 3-5 minutes)..."):
            start = time.time()
            try:
                result = process_image(img)
                if result:
                    st.image(result, caption="Ghibli Style")
                    buf = io.BytesIO()
                    result.save(buf, format="PNG")
                    st.download_button(
                        "Download Result",
                        buf.getvalue(),
                        "ghibli_style.png",
                        "image/png"
                    )
                    st.success(f"Completed in {time.time()-start:.1f} seconds")
            except Exception as e:
                st.error(f"Conversion failed: {str(e)}")
                st.info("Please try with a smaller image (max 512x512)")

st.markdown("---")
st.warning("""
**Free Tier Limitations:**
- Max image size: 512x512 pixels
- Processing time: 3-5 minutes
- CPU-only processing
""")
