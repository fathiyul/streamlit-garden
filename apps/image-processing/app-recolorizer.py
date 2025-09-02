import io
from typing import Tuple

import numpy as np
from PIL import Image, ImageOps
import streamlit as st
from streamlit_image_comparison import image_comparison

st.set_page_config(
    page_title="Two-Color Image Recolorizer", page_icon="🎨", layout="centered"
)

st.title("🎨 Two-Color Image Recolorizer")
st.write(
    "Upload an image to convert it to grayscale and map it to a two-color gradient."
)

# --- Sidebar controls
st.sidebar.header("Palette & Options")
color1 = st.sidebar.color_picker(
    "Color A (dark/low tones)", value="#22C55E"
)  # green-500
color2 = st.sidebar.color_picker(
    "Color B (light/high tones)", value="#EC4899"
)  # pink-500

reverse = st.sidebar.checkbox("Reverse mapping (swap A/B)", value=False)

# Per-channel gamma lets you tweak grayscale contrast before colorizing
gamma = st.sidebar.slider(
    "Grayscale gamma",
    min_value=0.2,
    max_value=3.0,
    value=1.0,
    step=0.05,
    help="<1 brightens midtones; >1 darkens midtones before mapping colors.",
)

# Intensity blend between the colorized result and the original grayscale
blend = st.sidebar.slider(
    "Color intensity",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="0 = grayscale only, 1 = fully colorized.",
)

st.sidebar.caption(
    "Tip: Use a darker color for A and a lighter color for B for natural gradients."
)

uploaded = st.file_uploader(
    "Upload an image", type=["png", "jpg", "jpeg", "webp", "bmp", "tiff"]
)


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join([c * 2 for c in hex_color])
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return r, g, b


def _apply_two_color_map(
    gray_img: Image.Image,
    c1: Tuple[int, int, int],
    c2: Tuple[int, int, int],
    gamma: float = 1.0,
    intensity: float = 1.0,
) -> Image.Image:
    """Map a single-channel grayscale image (0..255) to a linear gradient between c1 and c2.

    gamma: apply gamma on normalized grayscale before mapping.
    intensity: blend between grayscale (0) and full color (1).
    """
    arr = np.asarray(gray_img).astype(np.float32)
    if arr.ndim == 3:  # safety if grayscale came out as (H,W,3)
        arr = arr[..., 0]

    # Normalize and apply gamma
    t = np.clip(arr / 255.0, 0.0, 1.0)
    if gamma != 1.0:
        t = np.power(t, gamma)

    # Linear interpolation between colors
    c1 = np.array(c1, dtype=np.float32)
    c2 = np.array(c2, dtype=np.float32)
    rgb = (1.0 - t)[..., None] * c1 + t[..., None] * c2

    # Blend with grayscale to control intensity
    gray_rgb = np.repeat(arr[..., None], 3, axis=2)
    out = (1.0 - intensity) * gray_rgb + intensity * rgb
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


if uploaded is not None:
    with Image.open(uploaded) as img_in:
        img_in = img_in.convert("RGB")

    # Grayscale
    gray = ImageOps.grayscale(img_in)

    cA = _hex_to_rgb(color2 if reverse else color1)
    cB = _hex_to_rgb(color1 if reverse else color2)

    colored = _apply_two_color_map(gray, cA, cB, gamma=gamma, intensity=blend)

    st.subheader("Colorized (Two-Color Palette)")
    st.image(colored, width="stretch")

    # Download
    buf = io.BytesIO()
    colored.save(buf, format="PNG")
    buf.seek(0)

    filename = uploaded.name
    name, ext = filename.rsplit(".", 1)
    download_name = f"{name}_recolorized.{ext}"

    st.download_button(
        label="Download colorized image",
        data=buf,
        file_name=download_name,
        mime="image/png",
    )

    # Optional: Show grayscale separately if you want
    with st.expander("View intermediate steps"):
        # Before/After comparison
        st.subheader("Before vs After Comparison")
        image_comparison(
            img1=img_in,
            img2=colored,
            label1="Original",
            label2="Recolorized",
            width=700,
            starting_position=50,
            show_labels=True,
            make_responsive=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Original**")
            st.image(img_in, width="stretch")
        with col2:
            st.write("**Grayscale**")
            st.image(gray, width="stretch")

else:
    st.info("Upload an image to begin.")

st.markdown(
    "---\n"
    "**How it works:** We convert your image to grayscale, normalize brightness to 0–255,\n"
    "optionally apply a gamma curve, then blend linearly from Color A (for dark areas) to\n"
    "Color B (for bright areas)."
)
