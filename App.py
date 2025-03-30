import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline
import io
import time

# ========== KONFIGURASI WAJIB ==========
st.set_page_config(
    page_title="Ghibli Converter Free",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ========== MODEL LOADING ==========
@st.cache_resource(ttl=3600)  # Cache model untuk 1 jam
def load_model():
    try:
        # Gunakan model yang lebih ringan
        model_id = "nitrosocke/mo-di-diffusion"  # Alternatif lebih stabil
        
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,  # Wajib float32 untuk CPU
            safety_checker=None,  # Nonaktifkan safety checker
            requires_safety_checker=False
        )
        
        # WAJIB untuk Free Tier:
        pipe = pipe.to("cpu")
        pipe.enable_attention_slicing(1)
        pipe.enable_sequential_cpu_offload()
        
        return pipe
    except Exception as e:
        st.error(f"GAGAL MEMUAT MODEL: {str(e)}")
        return None

# ========== FUNGSI UTAMA ==========
def process_image(image):
    try:
        # Resize dengan aspect ratio
        max_size = 384  # Lebih kecil untuk free tier
        width, height = image.size
        ratio = max_size / max(width, height)
        new_size = (int(width * ratio), int(height * ratio))
        image = image.resize(new_size, Image.LANCZOS)
        
        pipe = load_model()
        if pipe is None:
            return None
            
        return pipe(
            prompt="ghibli style, studio ghibli, anime masterpiece",
            image=image,
            strength=0.5,  # Lebih rendah untuk CPU
            guidance_scale=6,
            num_inference_steps=15,  # Minimal steps
            generator=torch.Generator("cpu").manual_seed(42)
        ).images[0]
    except Exception as e:
        st.error(f"ERROR PROSES: {str(e)}")
        return None

# ========== INTERFACE ==========
st.title("🎨 100% WORK Ghibli Converter")
st.warning("⚠️ Untuk Streamlit Free Tier - CPU Only")

uploaded_file = st.file_uploader(
    "UNGGAH GAMBAR (MAX 384px)", 
    type=["jpg", "png"],
    help="Gambar portrait/landscape jelas bekerja lebih baik"
)

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Original", use_column_width=True)
    
    if st.button("✨ TRANSFORM", type="primary"):
        with st.spinner("PROSES (3-7 menit)..."):
            start_time = time.time()
            
            result = process_image(img)
            
            if result:
                st.image(result, caption="Ghibli Style", use_column_width=True)
                
                # Download handler
                buf = io.BytesIO()
                result.save(buf, format="PNG")
                
                st.download_button(
                    "💾 DOWNLOAD",
                    buf.getvalue(),
                    "ghibli_art.png",
                    "image/png"
                )
                
                st.success(f"✅ SELESAI! Waktu: {time.time()-start_time:.1f} detik")
            else:
                st.error("Gagal memproses. Coba gambar lain atau refresh halaman.")

# ========== FOOTER ==========
st.markdown("---")
st.caption("""
🔧 **Technical Specs:**
- CPU-only mode
- Max dimension: 384px
- Simplified model architecture
- Safety checks disabled
""")
