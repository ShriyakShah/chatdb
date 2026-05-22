import os
from flask import Flask, render_template, request, jsonify
from datetime import datetime
from dotenv import load_dotenv
import requests

load_dotenv()

app = Flask(__name__)

SITE_URL = os.environ.get("SITE_URL", "http://localhost:5000")
SITE_NAME = "AI Camera OCR & Saver"

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

    # --- STEP 1: Process Image via OCR.space ---
    try:
        image_file.seek(0)
        
        payload = {
            'apikey': ocr_space_key,
            'language': 'eng',
            'isOverlayRequired': False
        }
        
        files = {
            'file': (image_file.filename, image_file.read(), image_file.content_type)
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

    # --- STEP 2: Refine Extracted Text via OpenRouter (Direct HTTP Post) ---
    try:
        headers = {
            "Authorization": f"Bearer {openrouter_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": SITE_URL,
            "X-Title": SITE_NAME,
        }
        
        # Format prompt to clean up the raw OCR text
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
        
        # Extract the content from OpenRouter's response structure
        choices = ai_result.get("choices", [])
        if not choices:
            raise Exception("No choices returned from OpenRouter response.")
            
        refined_text = choices[0].get("message", {}).get("content", "").strip()
        return jsonify({'text': refined_text})

    except Exception as e:
        # Fallback to returning raw text if the OpenRouter AI call encounters an issue
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