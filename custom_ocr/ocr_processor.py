import json
from typing import Dict, Any, List, Union
import os
import base64
import requests
from tqdm import tqdm
import concurrent.futures
from pathlib import Path
import cv2
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import numpy as np

class OCRProcessor:
    def __init__(self, model_name: str = "llama3.2-vision:11b", 
                 base_url: str = "http://localhost:11434/api/generate",
                 max_workers: int = 1,
                 language: str = "auto",
                 tesseract_language: str = "spa"):
        
        self.model_name = model_name
        self.base_url = base_url
        self.max_workers = max_workers
        self.language = language
        self.tesseract_language = tesseract_language

    def _encode_image(self, image_path: str) -> str:
        """Convert image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _preprocess_image(self, image_path: str) -> str:
        """
        Preprocess image before OCR:
        - Convert PDF to image if needed
        - Auto-rotate
        - Enhance contrast
        - Reduce noise
        """
        # Handle PDF files
        if image_path.lower().endswith('.pdf'):
            pages = convert_from_path(image_path)
            if not pages:
                raise ValueError("Could not convert PDF to image")
            # Save first page as temporary image
            temp_path = f"{image_path}_temp.jpg"
            pages[0].save(temp_path, 'JPEG')
            image_path = temp_path

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced)

        # Save preprocessed image
        preprocessed_path = f"{image_path}_preprocessed.jpg"
        cv2.imwrite(preprocessed_path, denoised)

        return preprocessed_path

    def _preprocess_image_tesseract(self, image_path):
        """Preprocesar la imagen para mejorar la calidad del OCR"""
        img = cv2.imread(image_path)
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Binarización adaptativa
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Reducción de ruido
        denoised = cv2.fastNlMeansDenoising(binary)
        
        return denoised

    def _extract_text_tesseract(self, image):
        """Extraer texto usando Tesseract"""
        try:
            custom_config = f'--oem 3 --psm 6 -l {self.tesseract_language}'
            return pytesseract.image_to_string(image, config=custom_config)
        except Exception as e:
            print(f"Error en Tesseract: {str(e)}")
            return None

    def _verify_with_vision(self, image_path, text_tesseract):
        """Verificar y corregir el texto usando modelo de visión"""
        try:
            with open(image_path, "rb") as img:
                image_base64 = base64.b64encode(img.read()).decode()
            
            system_prompt = """Eres un sistema de OCR especializado en corrección de texto.
            INSTRUCCIONES ESTRICTAS:
            1. NO añadas texto introductorio
            2. NO agregues texto que no esté en la imagen
            3. NO traduzcas ni modifiques el contenido
            4. Mantén EXACTAMENTE el formato original
            5. Corrige errores de OCR comparando con la imagen
            6. Preserva números de página y espaciado"""

            user_prompt = f"""TEXTO A CORREGIR:
            {text_tesseract}
            
            Corrige los errores de OCR manteniendo el formato original."""

            response = requests.post(
                self.base_url,
                json={
                    "model": self.model_name,
                    "system": system_prompt,
                    "prompt": user_prompt,
                    "stream": False,
                    "images": [image_base64],
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 1024
                    }
                }
            )
            
            return response.json()["response"].strip()
        except Exception as e:
            print(f"Error en verificación con Vision: {str(e)}")
            return None

    def _clean_text(self, text):
        """Limpiar texto de caracteres extraños y patrones no deseados"""
        if not text:
            return text
            
        lines = text.split('\n')
        
        # Limpiar primera línea si contiene caracteres extraños
        if lines and any(c in lines[0] for c in '€<>—áAX'):
            # Intentar extraer número si existe
            import re
            numbers = re.findall(r'\d+', lines[0])
            if numbers:
                lines[0] = numbers[0]
            else:
                lines[0] = ""
        
        return '\n'.join(lines)

    def _calculate_metrics(self, original_text, extracted_text):
        """Calcular métricas de similitud entre textos"""
        from difflib import SequenceMatcher
        
        # Normalizar textos
        text1 = ' '.join(original_text.split())
        text2 = ' '.join(extracted_text.split())
        
        # Calcular ratio de similitud
        ratio = SequenceMatcher(None, text1, text2).ratio()
        
        # Contar palabras
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Calcular intersección
        common_words = words1.intersection(words2)
        word_accuracy = len(common_words) / max(len(words1), len(words2))
        
        return {
            'similarity_ratio': ratio * 100,
            'word_accuracy': word_accuracy * 100,
            'common_words': len(common_words),
            'total_words_original': len(words1),
            'total_words_extracted': len(words2)
        }

    def _get_language_prompt(self, format_type: str) -> str:
        """Get prompt template based on language and format"""
        prompts = {
            "es": {
                "markdown": """INSTRUCCIÓN: Transcribe LITERALMENTE el texto de la imagen.
                NO hagas lo siguiente:
                - NO describas la imagen
                - NO traduzcas el texto
                - NO interpretes el contenido
                - NO agregues comentarios

                SOLO haz esto:
                1. Lee el texto de la imagen
                2. Transcríbelo EXACTAMENTE como aparece
                3. Usa el formato markdown:
                   - # para títulos principales
                   - ## para subtítulos
                   - - para listas
                   - Respeta los párrafos originales

                EJEMPLO de lo que debes hacer:
                Si la imagen muestra:
                "Capítulo 1
                 El inicio
                 Había una vez..."
                
                Debes transcribir exactamente:
                # Capítulo 1
                ## El inicio
                Había una vez...""",

                "text": """Por favor, analiza esta imagen y extrae todo el contenido de texto.
                Proporciona la salida como texto plano, manteniendo el diseño original y los saltos de línea donde corresponda.
                Incluye todo el texto visible de la imagen.
                IMPORTANTE: Transcribe el texto EXACTAMENTE como aparece, sin traducir ni modificar.""",

                "json": """Por favor, analiza esta imagen y extrae todo el contenido de texto. Estructura la salida como JSON con estas pautas:
                - Identifica diferentes secciones o componentes
                - Usa claves apropiadas para diferentes elementos de texto
                - Mantén la estructura jerárquica del contenido
                - Incluye todo el texto visible de la imagen
                - IMPORTANTE: Transcribe el texto EXACTAMENTE como aparece, sin traducir ni modificar""",

                "structured": """Por favor, analiza esta imagen y extrae todo el contenido de texto, enfocándote en elementos estructurales:
                - Identifica y formatea cualquier tabla
                - Extrae listas y mantén su estructura
                - Preserva cualquier relación jerárquica
                - Formatea secciones y subsecciones claramente
                - IMPORTANTE: Transcribe el texto EXACTAMENTE como aparece, sin traducir ni modificar""",

                "key_value": """Por favor, analiza esta imagen y extrae el texto que aparece en pares clave-valor:
                - Busca etiquetas y sus valores asociados
                - Extrae campos de formulario y sus contenidos
                - Identifica cualquier información emparejada
                - Presenta cada par en una nueva línea como 'clave: valor'
                - IMPORTANTE: Transcribe el texto EXACTAMENTE como aparece, sin traducir ni modificar"""
            },
            "en": {
                "markdown": """Please look at this image and extract all the text content. Format the output in markdown:
                - Use headers (# ## ###) for titles and sections
                - Use bullet points (-) for lists
                - Use proper markdown formatting for emphasis and structure
                - Preserve the original text hierarchy and formatting""",

                "text": """Please look at this image and extract all the text content. 
                Provide the output as plain text, maintaining the original layout and line breaks where appropriate.
                Include all visible text from the image.""",

                "json": """Please look at this image and extract all the text content. Structure the output as JSON with these guidelines:
                - Identify different sections or components
                - Use appropriate keys for different text elements
                - Maintain the hierarchical structure of the content
                - Include all visible text from the image""",

                "structured": """Please look at this image and extract all the text content, focusing on structural elements:
                - Identify and format any tables
                - Extract lists and maintain their structure
                - Preserve any hierarchical relationships
                - Format sections and subsections clearly""",

                "key_value": """Please look at this image and extract text that appears in key-value pairs:
                - Look for labels and their associated values
                - Extract form fields and their contents
                - Identify any paired information
                - Present each pair on a new line as 'key: value'"""
            }
        }

        # Si el idioma es "auto", intentamos detectar basándonos en el contenido
        if self.language == "auto":
            # Por ahora, simplemente usamos español como predeterminado
            # TODO: Implementar detección automática de idioma
            return prompts["es"].get(format_type, prompts["es"]["text"])
        
        # Si el idioma está especificado, usamos ese
        return prompts.get(self.language, prompts["en"]).get(format_type, prompts["en"]["text"])

    def process_image(self, image_path: str, format_type: str = "markdown", preprocess: bool = True) -> str:
        """
        Process an image and extract text in the specified format
        
        Args:
            image_path: Path to the image file
            format_type: One of ["markdown", "text", "json", "structured", "key_value"]
            preprocess: Whether to apply image preprocessing
        """
        try:
            if preprocess:
                image_path = self._preprocess_image(image_path)
            
            image_base64 = self._encode_image(image_path)
            
            # Clean up temporary files
            if image_path.endswith(('_preprocessed.jpg', '_temp.jpg')):
                os.remove(image_path)

            # Crear un prompt más específico con instrucciones en español
            system_prompt = """IMPORTANTE: Eres un transcriptor LITERAL de texto.
            
            REGLAS ESTRICTAS:
            1. SOLO transcribe el texto que VES en la imagen
            2. Si no puedes leer algo, pon [...] y continúa
            3. NO INVENTES texto que no está en la imagen
            4. NO REPITAS texto
            5. NO AGREGUES nada extra
            6. Transcribe línea por línea, exactamente como aparece
            
            Si no estás 100% seguro de ver algo, NO lo transcribas."""

            # Get the appropriate prompt based on language
            content_prompt = "TRANSCRIBE EL TEXTO DE LA IMAGEN EXACTAMENTE COMO ESTÁ, LÍNEA POR LÍNEA:"

            # Prepare the request payload with both prompts
            payload = {
                "model": self.model_name,
                "system": system_prompt,
                "prompt": content_prompt,
                "stream": False,
                "images": [image_base64],
                "options": {
                    "temperature": 0.0,  # Hacer el modelo completamente determinista
                    "num_predict": 500,  # Limitar la salida para evitar alucinaciones
                    "stop": ["[fin]", "[FIN]", "FIN", "Fin"]  # Intentar detener la generación
                }
            }

            # Make the API call to Ollama
            response = requests.post(self.base_url, json=payload)
            response.raise_for_status()
            
            result = response.json().get("response", "")
            
            # Clean up the result if needed
            if format_type == "json":
                try:
                    json_data = json.loads(result)
                    return json.dumps(json_data, indent=2)
                except json.JSONDecodeError:
                    return result
            
            return result
        except Exception as e:
            return f"Error processing image: {str(e)}"

    def process_image_tesseract(self, image_path: str, format_type: str = "markdown", preprocess: bool = True) -> str:
        """
        Process an image and extract text in the specified format using Tesseract
        
        Args:
            image_path: Path to the image file
            format_type: One of ["markdown", "text", "json", "structured", "key_value"]
            preprocess: Whether to apply image preprocessing
        """
        try:
            if preprocess:
                image = self._preprocess_image_tesseract(image_path)
            else:
                image = cv2.imread(image_path)
            
            text_tesseract = self._extract_text_tesseract(image)
            if text_tesseract is None:
                raise Exception("Error en la extracción de texto con Tesseract")
            
            text_tesseract = self._clean_text(text_tesseract)
            
            text_verified = self._verify_with_vision(image_path, text_tesseract)
            if text_verified is None:
                raise Exception("Error en la verificación con Vision")
            
            text_verified = self._clean_text(text_verified)
            
            result = {
                'text_tesseract': text_tesseract,
                'text_final': text_verified
            }
            
            return result
            
        except Exception as e:
            return {'error': str(e)}

    def process_batch(
        self,
        input_path: Union[str, List[str]],
        format_type: str = "markdown",
        recursive: bool = False,
        preprocess: bool = True
    ) -> Dict[str, Any]:
        """
        Process multiple images in batch
        
        Args:
            input_path: Path to directory or list of image paths
            format_type: Output format type
            recursive: Whether to search directories recursively
            preprocess: Whether to apply image preprocessing
            
        Returns:
            Dictionary with results and statistics
        """
        # Collect all image paths
        image_paths = []
        if isinstance(input_path, str):
            base_path = Path(input_path)
            if base_path.is_dir():
                pattern = '**/*' if recursive else '*'
                for ext in ['.png', '.jpg', '.jpeg', '.pdf', '.tiff']:
                    image_paths.extend(base_path.glob(f'{pattern}{ext}'))
            else:
                image_paths = [base_path]
        else:
            image_paths = [Path(p) for p in input_path]

        results = {}
        errors = {}
        
        # Process images in parallel with progress bar
        with tqdm(total=len(image_paths), desc="Processing images") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_path = {
                    executor.submit(self.process_image_tesseract, str(path), format_type, preprocess): path
                    for path in image_paths
                }
                
                for future in concurrent.futures.as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        results[str(path)] = future.result()
                    except Exception as e:
                        errors[str(path)] = str(e)
                    pbar.update(1)

        return {
            "results": results,
            "errors": errors,
            "statistics": {
                "total": len(image_paths),
                "successful": len(results),
                "failed": len(errors)
            }
        }