import os
import io
from flask import Flask, render_template, request, jsonify
from datetime import datetime
from dotenv import load_dotenv
import requests
from PIL import Image

load_dotenv()

app = Flask(__name__)

SITE_URL = os.environ.get("SITE_URL", "http://localhost:5000")
SITE_NAME = "AI Camera OCR & Saver"

def compress_image_to_limit(image_file, max_size_bytes=1000000):
    """
    Opens an image using Pillow, scales it down to maximum 2000px dimension
    (which is optimal for OCR), and compresses it to a JPEG under 1MB.
    """
    image_file.seek(0)
    img = Image.open(image_file)
    
    # Convert RGBA/Palette images to standard RGB (JPEGs don't support alpha transparency)
    if img.mode in ('RGBA', 'P', 'LA'):
        img = img.convert('RGB')
    
    # Scale down dimensions if the photo is extremely large (e.g. 4000px+ from high-end cameras)
    # 2000px is more than enough resolution for OCR processing
    max_dimension = 2000
    width, height = img.size
    if max(width, height) > max_dimension:
        img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
    
    # Progressively test lower compression qualities until size is under 1MB
    for quality in [85, 70, 50, 30]:
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality, optimize=True)
        size = buffer.tell()
        if size <= max_size_bytes:
            buffer.seek(0)
            return buffer.getvalue(), "image/jpeg", "compressed_image.jpg"
            
    # Ultimate fallback: aggressively scale down image if still too large
    img.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=30, optimize=True)
    buffer.seek(0)
    return buffer.getvalue(), "image/jpeg", "compressed_image.jpg"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    # Verify environment variables
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    ocr_space_key = os.environ.get("OCR_SPACE_API_KEY")
    
    if not openrouter_key:
        return jsonify({'error': 'OPENROUTER_API_KEY is missing in Render settings.'}), 500
    if not ocr_space_key:
        return jsonify({'error': 'OCR_SPACE_API_KEY is missing in Render settings.'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    raw_text = ""

    # --- STEP 1: Process Image (with compression fallback) ---
    try:
        # Determine the raw file size in bytes
        image_file.seek(0, os.SEEK_END)
        file_size = image_file.tell()
        image_file.seek(0)

        # If the file is larger than 1MB, compress it
        if file_size > 1000000:
            file_content, content_type, filename = compress_image_to_limit(image_file)
        else:
            file_content = image_file.read()
            content_type = image_file.content_type
            filename = image_file.filename

        # Send image data to OCR.space API
        payload = {
            'apikey': ocr_space_key,
            'language': 'eng',
            'isOverlayRequired': False
        }
        
        files = {
            'file': (filename, file_content, content_type)
        }
        
        ocr_response = requests.post(
            'https://api.ocr.space/parse/image',
            files=files,
            data=payload,
            timeout=30
        )
        
        if ocr_response.status_code != 200:
            return jsonify({'error': f"OCR.space returned HTTP status {ocr_response.status_code}"}), 500
        
        ocr_result = ocr_response.json()
        
        if ocr_result.get("IsErroredOnProcessing"):
            error_message = ocr_result.get("ErrorMessage", ["Unknown processing error"])[0]
            return jsonify({'error': f"OCR.space Error: {error_message}"}), 500
        
        parsed_results = ocr_result.get("ParsedResults", [])
        if not parsed_results:
            return jsonify({'error': 'OCR.space did not return structured parsing results.'}), 400
        
        raw_text = parsed_results[0].get("ParsedText", "").strip()
        if not raw_text:
            return jsonify({'error': 'No text detected in the image.'}), 400

    except Exception as e:
        return jsonify({'error': f"OCR processing failed: {str(e)}"}), 500

    # --- STEP 2: Refine Extracted Text via OpenRouter ---
    try:
        headers = {
            "Authorization": f"Bearer {openrouter_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": SITE_URL,
            "X-Title": SITE_NAME,
        }
        
        prompt = (
            "The following text is raw output from an OCR tool. Please clean it up, "
            "correct formatting, spacing, grammar, and spelling mistakes, and return only "
            "the final cleaned text. Do not write any introduction, commentary, or extra explanations.\n\n"
            f"Raw text:\n{raw_text}"
        )
        
        json_data = {
            "model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        ai_response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=json_data,
            timeout=30
        )
        
        if ai_response.status_code != 200:
            raise Exception(f"OpenRouter API returned HTTP status {ai_response.status_code}")
            
        ai_result = ai_response.json()
        choices = ai_result.get("choices", [])
        if not choices:
            raise Exception("No choices returned from OpenRouter response.")
            
        refined_text = choices[0].get("message", {}).get("content", "").strip()
        return jsonify({'text': refined_text})

    except Exception as e:
        return jsonify({
            'text': f"[AI Cleanup failed. Presenting raw text fallback]\n\n{raw_text}",
            'warning': f"OpenRouter refinement failed: {str(e)}"
        })

@app.route('/save', methods=['POST'])
def save_text():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided to save'}), 400

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"saved_text_{timestamp}.txt"
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(current_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
            
        return jsonify({'success': True, 'filename': filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)