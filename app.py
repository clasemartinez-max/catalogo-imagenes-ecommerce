import io
import zipfile

import numpy as np
import streamlit as st
from PIL import Image
from rembg import new_session, remove

PLATFORM_SPECS = {
    "Google Shopping": {"size": (1200, 1200), "fill": 0.85},
    "Mercado Libre": {"size": (1200, 1200), "fill": 0.85},
    "TiendaNube": {"size": (1200, 1200), "fill": 0.85},
    "Pinterest": {"size": (1000, 1500), "fill": 0.80},
    "Instagram": {"size": (1080, 1080), "fill": 0.85},
    "Otra plataforma / tamaño original": {"size": None, "fill": 0.85},
}

BG_COLORS = {
    "Blanco": (255, 255, 255, 255),
    "Gris claro": (242, 242, 242, 255),
    "Transparente": (0, 0, 0, 0),
}


@st.cache_resource
def get_session_general():
    return new_session("isnet-general-use")


@st.cache_resource
def get_session_human():
    return new_session("u2net_human_seg")


def remove_background(raw_bytes: bytes, quitar_mano: bool, umbral_humano: int = 50) -> Image.Image:
    session_general = get_session_general()

    mask_general_bytes = remove(
        raw_bytes,
        session=session_general,
        only_mask=True,
        post_process_mask=True,
    )
    mask_general = Image.open(io.BytesIO(mask_general_bytes)).convert("L")

    if quitar_mano:
        session_human = get_session_human()
        mask_human_bytes = remove(
            raw_bytes,
            session=session_human,
            only_mask=True,
            post_process_mask=True,
        )
        mask_human = Image.open(io.BytesIO(mask_human_bytes)).convert("L")

        arr_general = np.array(mask_general)
        arr_human = np.array(mask_human)
        arr_final = np.where(arr_human > umbral_humano, 0, arr_general).astype("uint8")

        # Fallback: si la resta dejó la máscara casi vacía, usar la general
        if arr_final.sum() < arr_general.sum() * 0.05:
            mask_final = mask_general
        else:
            mask_final = Image.fromarray(arr_final, mode="L")
    else:
        mask_final = mask_general

    original = Image.open(io.BytesIO(raw_bytes)).convert("RGBA")
    original.putalpha(mask_final)
    return original


def compose_on_canvas(img_rgba: Image.Image, platform: str, bg_color_name: str) -> Image.Image:
    spec = PLATFORM_SPECS[platform]
    canvas_size = spec["size"]
    fill = spec["fill"]
    bg_color = BG_COLORS[bg_color_name]

    bbox = img_rgba.getbbox()
    if bbox:
        img_rgba = img_rgba.crop(bbox)

    if canvas_size is None:
        # Tamaño original: solo cambiar fondo
        if bg_color_name == "Transparente":
            return img_rgba
        canvas = Image.new("RGBA", img_rgba.size, bg_color)
        canvas.paste(img_rgba, (0, 0), img_rgba)
        return canvas

    max_w = int(canvas_size[0] * fill)
    max_h = int(canvas_size[1] * fill)
    img_rgba.thumbnail((max_w, max_h), Image.LANCZOS)

    if bg_color_name == "Transparente":
        canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    else:
        canvas = Image.new("RGBA", canvas_size, bg_color)

    x = (canvas_size[0] - img_rgba.width) // 2
    y = (canvas_size[1] - img_rgba.height) // 2
    canvas.paste(img_rgba, (x, y), img_rgba)
    return canvas


def image_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def resize_for_preview(img: Image.Image, max_size: int = 300) -> Image.Image:
    preview = img.copy()
    preview.thumbnail((max_size, max_size), Image.LANCZOS)
    return preview


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Limpieza de fondo para catálogo", layout="wide")

st.title("Limpieza de fondo para catálogo de e-commerce")
st.caption("Eliminá el fondo de tus fotos de producto y exportalas listas para cada plataforma.")

with st.sidebar:
    st.header("Configuración")
    platform = st.selectbox("Plataforma de destino", list(PLATFORM_SPECS.keys()))
    bg_color_name = st.selectbox("Color de fondo", list(BG_COLORS.keys()))
    quitar_mano = st.checkbox(
        "Las fotos tienen una mano sosteniendo el producto (quitar mano/brazo)",
        value=False,
        help="Activa un segundo modelo de segmentación. Más lento pero elimina piel/manos.",
    )
    st.divider()
    spec = PLATFORM_SPECS[platform]
    if spec["size"]:
        st.caption(f"Canvas: {spec['size'][0]}×{spec['size'][1]} px · Relleno: {int(spec['fill']*100)}%")
    else:
        st.caption("Se conserva el tamaño original de cada imagen.")

uploaded_files = st.file_uploader(
    "Subí las fotos de producto",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
)

if uploaded_files:
    if len(uploaded_files) > 30:
        st.info(
            f"Subiste {len(uploaded_files)} imágenes. El lote es grande: puede tardar varios minutos "
            "y consumir bastante memoria, especialmente en hosting gratuito."
        )

    if st.button("Procesar todas", type="primary"):
        resultados = {}
        errores = []
        progress = st.progress(0, text="Iniciando...")

        for i, file in enumerate(uploaded_files):
            progress.progress((i) / len(uploaded_files), text=f"Procesando {file.name}…")
            try:
                raw = file.read()
                img_sin_fondo = remove_background(raw, quitar_mano)
                img_final = compose_on_canvas(img_sin_fondo, platform, bg_color_name)
                nombre_salida = file.name.rsplit(".", 1)[0] + "_catalogo.png"
                resultados[nombre_salida] = image_to_bytes(img_final)
            except Exception as e:
                errores.append((file.name, str(e)))
                st.error(f"Error al procesar **{file.name}**: {e}")

        progress.progress(1.0, text="Listo.")

        if errores and not resultados:
            st.warning("No se pudo procesar ninguna imagen. Revisá los archivos e intentá de nuevo.")
        elif resultados:
            st.success(f"Se procesaron {len(resultados)} imagen(es) correctamente.")

            # Vista previa en grilla
            cols = st.columns(min(4, len(resultados)))
            for idx, (nombre, data) in enumerate(resultados.items()):
                img = Image.open(io.BytesIO(data))
                preview = resize_for_preview(img)
                cols[idx % 4].image(preview, caption=nombre, use_container_width=True)

            # ZIP en memoria
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for nombre, data in resultados.items():
                    zf.writestr(nombre, data)
            zip_buf.seek(0)

            st.download_button(
                label=f"Descargar todas ({len(resultados)} imágenes) — .zip",
                data=zip_buf,
                file_name="catalogo_procesado.zip",
                mime="application/zip",
            )

st.divider()
st.markdown(
    "🔒 **Privacidad:** las imágenes se procesan en memoria durante la sesión y no se almacenan en ningún servidor.  \n"
    "Plataforma creada por [**clasemartinez**](https://www.linkedin.com/in/claudiomartinez1/)"
)
