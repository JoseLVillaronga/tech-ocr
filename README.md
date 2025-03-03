# CPE Label OCR Project

## Descripción

Este proyecto tiene como objetivo desarrollar una solución para extraer información identificatoria y funcional a partir de imágenes (en formato PNG) de etiquetas de CPEs (Customer Premises Equipment). La idea es procesar las imágenes para extraer campos como serie, MAC, PON ID, password, y modelo, y devolver un JSON con un esquema fijo con estos datos.

El flujo actual utiliza dos componentes principales:

1. **OCR con Tesseract:**
   - Se realiza la extracción inicial del texto desde la imagen con Tesseract.
   - Se aplica un preprocesamiento de imagen (contraste, reducción de ruido, binarización) para mejorar la precisión.

2. **Verificación con un Modelo de Visión (llama3.2-vision:11b):**
   - Se envía el texto extraído y la imagen a un servicio de verificación.
   - El servicio corrige los errores del OCR para entregar un texto final limpio y en formato plano.

Una vez obtenido el texto final, se aplica un módulo de postprocesamiento que utiliza expresiones regulares para extraer los campos específicos.

## Estructura del Proyecto

- `/custom_ocr/ocr_processor.py`: Contiene la clase `OCRProcessor` que implementa todo el pipeline OCR, incluyendo preprocesamiento de la imagen, extracción de texto con Tesseract, y verificación con el modelo de visión.
- `/api.py`: Implementa la API con FastAPI que permite recibir imágenes PNG de etiquetas, procesarlas con el OCR y devolver un JSON con los datos extraídos.
- `/test_ocr_processor.py`: Archivo de pruebas que muestra cómo se invoca el pipeline OCR y se obtienen los resultados.
- `requirements.txt`: Lista las dependencias necesarias para el proyecto (Pillow, requests, pdf2image, pytesseract, opencv-python, tqdm, numpy, transformers, streamlit).
- `README.md`: Este archivo de documentación.

## Uso de la API

La API se ejecuta con **uvicorn**:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Para probar la API, puedes usar el siguiente comando **curl**:

```bash
curl -X POST -F "file=@/path/to/your/file.png" http://localhost:8000/extract
```

Este comando envía un archivo PNG a la ruta `/extract` y devuelve un JSON con los campos extraídos, que incluye:
- `serie`: Extraído a partir de la leyenda 'GPON SN:'.
- `mac`: La dirección MAC, sin separadores.
- `pon_id`, `password` y `modelo`.

## Sugerencias para Futuras Iteraciones

- Ajustar y optimizar los parámetros de preprocesamiento para adaptarse a variaciones en la imagen.
- Mejorar las expresiones regulares para capturar de forma robusta variaciones en la nomenclatura de los campos.
- Integrar la detección y lectura de códigos QR y de barras mediante bibliotecas como `pyzbar` si se presentan en las imágenes.
- Refinar la comunicación con el modelo de visión para obtener una salida de texto aún más precisa.

## Próximos Pasos

- Optimización del preprocesamiento y extracción OCR.
- Extender las pruebas unitarias e integradas para cubrir variaciones en las etiquetas.
- Integrar la extracción de QR/Barras para complementar los datos obtenidos.

---

Documentación elaborada para facilitar la transferencia de conocimientos tanto a desarrolladores como a sistemas de IA. Se recomienda mantener este documento actualizado con cada iteración del desarrollo.
