from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import os
import tempfile
import re

from custom_ocr.ocr_processor import OCRProcessor

app = FastAPI(
    title="CPE Label OCR API",
    description="Extrae datos identificatorios y funcionales de etiquetas CPE",
    version="1.0"
)

def extract_label_data(text: str) -> dict:
    """Extrae campos como serie, mac, pon_id, password y modelo del texto usando expresiones regulares."""
    patterns = {
        "serie": r"gpon sn[:\s]*([\w-]+)",
        "mac": r"mac[:\s]*([\da-fA-F:]+)",
        "pon_id": r"pon[_\s]?id[:\s]*([\w-]+)",
        "password": r"(clave de acceso al|contraseña|password)[:\s]*([\w-]+)",
        "modelo": r"modelo[:\s]*([\w\s-]+)"
    }
    data = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            data[key] = match.group(1).strip()
            # Eliminar separadores de la MAC
            if key == "mac":
                data[key] = data[key].replace(":", "")
        else:
            data[key] = ""
    return data


@app.post("/extract")
async def extract_data(file: UploadFile = File(...)):
    # Verificar que el archivo sea un PNG
    if file.content_type != "image/png":
        raise HTTPException(status_code=400, detail="Only PNG images are accepted")
    try:
        # Guardar el archivo subido en un archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(await file.read())
            tmp_filename = tmp.name

        # Instanciar el procesador OCR
        processor = OCRProcessor(tesseract_language="spa")
        # Se asume que process_image_tesseract procesa la imagen y retorna un diccionario con la llave 'text_final'
        result = processor.process_image_tesseract(tmp_filename, preprocess=True)
        text_final = result.get("text_final", "")

        # Eliminar el archivo temporal
        os.remove(tmp_filename)

        # Extraer datos de la etiqueta a partir del texto final
        extracted_data = extract_label_data(text_final)
        return JSONResponse(content=extracted_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
