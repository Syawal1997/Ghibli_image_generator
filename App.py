import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline
import io
import time

# Konfigurasi dasar
st.set_page_config(
    page_title="Ghibli Magic ✨",
    layout="centered",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def load_model():
    try:
        # Model yang lebih ringan untuk free tier
        model_id = "nitrosocke/Ghibli-Diffusion"
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        pipe.enable_attention_slicing()  # Mengurangi penggunaan VRAM
        return pipe
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        return None

def resize_image(image, max_size=512):
    width, height = image.size
    if max(width, height) > max_size:
        ratio = max_size / max(width, height)
        new_size = (int(width * ratio), int(height * ratio))
        return image.resize(new_size, Image.LANCZOS)
    return image

# UI
st.title("🎨 Ghibli-fy Your Photos!")
st.caption("Free Tier Version - Proses mungkin memakan waktu 2-5 menit")

uploaded_file = st.file_uploader(
    "Unggah gambar (JPEG/PNG)", 
    type=["jpg", "jpeg", "png"],
    help="Gambar dengan objek jelas bekerja paling baik"
)

if uploaded_file:
    col1, col2 = st.columns(2)
    with col1:
        original = Image.open(uploaded_file)
        st.image(original, caption="Original", use_column_width=True)
    
    with col2:
        if st.button("✨ Transform to Ghibli!", type="primary"):
            with st.spinner("Brewing some Studio Ghibli magic..."):
                start_time = time.time()
                
                try:
                    # Optimasi untuk free tier
                    small_image = resize_image(original, 512)
                    
                    pipe = load_model()
                    if pipe:
                        result = pipe(
                            prompt="ghibli style, high quality, vibrant colors",
                            image=small_image,
                            strength=0.65,  # Nilai balance untuk free tier
                            guidance_scale=7,
                            num_inference_steps=25  # Lebih rendah untuk free tier
                        ).images[0]
                        
                        st.image(result, caption="Ghibli Version", use_column_width=True)
                        
                        # Download
                        buf = io.BytesIO()
                        result.save(buf, format="PNG")
                        st.download_button(
                            "Download Result",
                            buf.getvalue(),
                            "ghibli_version.png",
                            "image/png"
                        )
                        
                        st.success(f"Done in {time.time()-start_time:.1f} seconds!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Free tier has limited resources. Try smaller images or refresh the app.")

# Catatan penting
st.markdown("---")
st.warning("""
**Free Tier Limitations:**
- Max image size: 512px
- Processing time: ~2-5 minutes
- May fail during peak hours
""")
