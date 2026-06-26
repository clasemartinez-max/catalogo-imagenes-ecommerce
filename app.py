import gc
import io
import zipfile

import numpy as np
import streamlit as st
from PIL import Image, ImageFilter
from rembg import new_session, remove

MAX_INPUT_PX = 1500  # límite antes de enviar a rembg — suficiente para canvas 1200×1200

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

MODELS = {
    "Productos generales": "isnet-general-use",
    "Ropa y textiles": "u2net_cloth_seg",
    "Fondo complejo o exterior": "u2net",
    "Objetos con bordes finos": "silueta",
}


@st.cache_resource
def get_session(model_name: str):
    return new_session(model_name)


@st.cache_resource
def get_session_human():
    return new_session("u2net_human_seg")


def preprocess(raw_bytes: bytes) -> bytes:
    """Reduce la imagen a MAX_INPUT_PX en su lado más largo para ahorrar RAM en rembg."""
    img = Image.open(io.BytesIO(raw_bytes))
    if max(img.size) > MAX_INPUT_PX:
        img.thumbnail((MAX_INPUT_PX, MAX_INPUT_PX), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    return raw_bytes


def remove_background(raw_bytes: bytes, model_name: str, quitar_mano: bool) -> Image.Image:
    raw_bytes = preprocess(raw_bytes)
    session = get_session(model_name)

    mask_general_bytes = remove(
        raw_bytes,
        session=session,
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

        # Restar píxeles con confianza de ser humanos >= 50% (128/255)
        arr_final = np.where(arr_human > 128, 0, arr_general).astype("uint8")

        # Fallback solo si la resta dejó menos del 5% de la máscara original (imagen casi vacía)
        if arr_final.sum() >= arr_general.sum() * 0.05:
            mask_general = Image.fromarray(arr_final, mode="L")

    original = Image.open(io.BytesIO(raw_bytes)).convert("RGBA")
    original.putalpha(mask_general)
    return original


def add_shadow(img_rgba: Image.Image, canvas_size: tuple, x: int, y: int) -> Image.Image:
    """Genera una sombra difusa debajo del producto y la compone sobre el canvas."""
    shadow_offset = max(8, canvas_size[0] // 120)
    blur_radius = max(12, canvas_size[0] // 80)

    # Extraer el canal alfa del producto y convertirlo en sombra gris oscuro
    alpha = img_rgba.split()[3]
    shadow_layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    shadow_color = Image.new("RGBA", img_rgba.size, (30, 30, 30, 110))
    shadow_color.putalpha(alpha)
    shadow_layer.paste(shadow_color, (x + shadow_offset, y + shadow_offset), shadow_color)
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(blur_radius))
    return shadow_layer


def compose_on_canvas(
    img_rgba: Image.Image, platform: str, bg_color_name: str, con_sombra: bool
) -> Image.Image:
    spec = PLATFORM_SPECS[platform]
    canvas_size = spec["size"]
    fill = spec["fill"]
    bg_color = BG_COLORS[bg_color_name]

    bbox = img_rgba.getbbox()
    if bbox:
        img_rgba = img_rgba.crop(bbox)

    if canvas_size is None:
        if bg_color_name == "Transparente":
            return img_rgba
        canvas = Image.new("RGBA", img_rgba.size, bg_color)
        canvas.paste(img_rgba, (0, 0), img_rgba)
        return canvas

    max_w = int(canvas_size[0] * fill)
    max_h = int(canvas_size[1] * fill)
    img_rgba.thumbnail((max_w, max_h), Image.LANCZOS)

    x = (canvas_size[0] - img_rgba.width) // 2
    y = (canvas_size[1] - img_rgba.height) // 2

    if bg_color_name == "Transparente":
        canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    else:
        canvas = Image.new("RGBA", canvas_size, bg_color)

    if con_sombra and bg_color_name != "Transparente":
        shadow = add_shadow(img_rgba, canvas_size, x, y)
        canvas = Image.alpha_composite(canvas, shadow)

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

# Inicializar session state
if "resultados" not in st.session_state:
    st.session_state.resultados = {}

def limpiar_sesion():
    st.session_state.resultados = {}
    gc.collect()

with st.sidebar:
    st.header("Configuración")
    platform = st.selectbox("Plataforma de destino", list(PLATFORM_SPECS.keys()))
    bg_color_name = st.selectbox("Color de fondo", list(BG_COLORS.keys()))
    model_label = st.selectbox(
        "Tipo de producto",
        list(MODELS.keys()),
        index=0,
        help="Elegí el tipo que mejor describa tus productos para obtener mejores resultados.",
    )
    model_name = MODELS[model_label]
    quitar_mano = st.checkbox(
        "Las fotos tienen una mano sosteniendo el producto (quitar mano/brazo)",
        value=False,
        help="Activa un segundo modelo de segmentación. Más lento pero elimina piel/manos.",
    )
    con_sombra = st.checkbox(
        "Agregar sombra sutil al producto",
        value=False,
        help="Añade una sombra difusa debajo del producto, estilo catálogo profesional.",
    )
    st.divider()
    spec = PLATFORM_SPECS[platform]
    if spec["size"]:
        st.caption(f"Canvas: {spec['size'][0]}×{spec['size'][1]} px · Relleno: {int(spec['fill']*100)}%")
    else:
        st.caption("Se conserva el tamaño original de cada imagen.")
    st.divider()
    if st.button("🗑️ Limpiar y empezar de nuevo", width="stretch"):
        limpiar_sesion()

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
        # Limpiar resultados anteriores antes de cada nueva corrida
        limpiar_sesion()

        errores = []
        progress = st.progress(0, text="Iniciando...")

        for i, file in enumerate(uploaded_files):
            progress.progress(i / len(uploaded_files), text=f"Procesando {file.name}…")
            try:
                raw = file.read()
                img_sin_fondo = remove_background(raw, model_name, quitar_mano)
                img_final = compose_on_canvas(img_sin_fondo, platform, bg_color_name, con_sombra)
                nombre_salida = file.name.rsplit(".", 1)[0] + "_catalogo.png"
                full_bytes = image_to_bytes(img_final)
                preview_bytes = image_to_bytes(resize_for_preview(img_final))
                st.session_state.resultados[nombre_salida] = {
                    "full": full_bytes,
                    "preview": preview_bytes,
                }
                del img_sin_fondo, img_final, raw, full_bytes, preview_bytes
                gc.collect()
            except Exception as e:
                errores.append((file.name, str(e)))
                st.error(f"Error al procesar **{file.name}**: {e}")

        progress.progress(1.0, text="Listo.")

        if errores and not st.session_state.resultados:
            st.warning("No se pudo procesar ninguna imagen. Revisá los archivos e intentá de nuevo.")

if st.session_state.resultados:
    st.success(f"Se procesaron {len(st.session_state.resultados)} imagen(es) correctamente.")

    cols = st.columns(min(4, len(st.session_state.resultados)))
    for idx, (nombre, entry) in enumerate(st.session_state.resultados.items()):
        cols[idx % 4].image(entry["preview"], caption=nombre, width="stretch")

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for nombre, entry in st.session_state.resultados.items():
            zf.writestr(nombre, entry["full"])
    zip_buf.seek(0)

    st.download_button(
        label=f"Descargar todas ({len(st.session_state.resultados)} imágenes) — .zip",
        data=zip_buf,
        file_name="catalogo_procesado.zip",
        mime="application/zip",
    )

st.divider()
st.markdown(
    "🔒 **Privacidad:** las imágenes se procesan en memoria durante la sesión y no se almacenan en ningún servidor.  \n"
    "Plataforma creada por [**clasemartinez**](https://www.linkedin.com/in/claudiomartinez1/)"
)
