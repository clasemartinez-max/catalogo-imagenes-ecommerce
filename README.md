# Limpieza de fondo para catálogo de e-commerce

App de Streamlit que recibe fotos de producto en lote, elimina el fondo (incluyendo manos que sostengan el producto) y devuelve las imágenes centradas sobre un canvas blanco, gris o transparente, con el tamaño y proporción correctos para la plataforma de venta elegida.

Plataformas soportadas: Google Shopping, Mercado Libre, TiendaNube, Pinterest, Instagram, o tamaño original.

---

## Limitación conocida — eliminación de manos

La opción "quitar mano/brazo" usa una técnica de resta de dos máscaras (`isnet-general-use` menos `u2net_human_seg`). Funciona bien en el caso típico (mano sosteniendo el producto desde abajo o el costado, con el producto mayormente visible). Si los dedos rodean completamente el producto y lo tapan en partes, pueden quedar bordes irregulares justo en la zona de contacto. No promete resultados perfectos al 100% en todos los casos.

---

## Cómo correrla localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requiere Python 3.11+.

---


Creado por [**clasemartinez**](https://www.linkedin.com/in/claudiomartinez1/)
