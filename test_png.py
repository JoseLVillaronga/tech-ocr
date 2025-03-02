from ollama_ocr import OCRProcessor

# Initialize OCR processor
ocr = OCRProcessor(model_name='llama3.2-vision:11b')  # Volvemos al modelo anterior que daba mejores resultados

# Process an image
result = ocr.process_image(
    image_path="/home/jose/Ollama-OCR/test-texto.png",
    format_type="markdown",
    preprocess=True
)
print(result)
