from custom_ocr.ocr_processor import OCRProcessor

# Texto original para comparación
texto_original = """159

para vehículos terrestres, cinco nativos humanoides salieron con solemne dignidad por una ventana del segundo piso.

Era una ventana de doble hoja que podía acomodar con facilidad tal procesión. El umbral y el dintel estaban decorados con treinta o cuarenta calaveras de aspecto humano. Luis no logró distinguir ningún orden especial en su distribución.

Los cinco se dirigieron hacia las aerocicletas. Cuando estuvieron cerca titubearon un momento; sin duda, intentaban decidir quién sería el jefe. Esos nativos también tenían aspecto humano, aunque no del todo. Era evidente que no pertenecían a ninguna raza humana conocida."""

# Crear procesador OCR
processor = OCRProcessor(
    model_name="llama3.2-vision:11b",
    tesseract_language="spa"
)

# Procesar imagen
result = processor.process_image_tesseract(
    "test-texto.png",
    preprocess=True
)

# Mostrar resultados
print("\nTexto extraído por Tesseract:")
print("-" * 50)
print(result['text_tesseract'])

print("\nTexto verificado por Vision:")
print("-" * 50)
print(result['text_final'])

# Calcular métricas
metrics = processor._calculate_metrics(texto_original, result['text_final'])
print("\nMétricas de similitud:")
print("-" * 50)
print(f"Ratio de similitud: {metrics['similarity_ratio']:.2%}")
print(f"Precisión de palabras: {metrics['word_accuracy']:.2%}")
print(f"Palabras en común: {metrics['common_words']}")
print(f"Palabras en original: {metrics['total_words_original']}")
print(f"Palabras extraídas: {metrics['total_words_extracted']}")
