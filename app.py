import os
import base64
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv

# Load a local .env file if it exists
load_dotenv()

app = Flask(__name__)

# Replace with your production domain after setting it up
SITE_URL = os.environ.get("SITE_URL", "http://localhost:5000")
SITE_NAME = "AI Camera OCR & Saver"

def get_openai_client():
    # Fetch OpenRouter key or standard key securely
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # Fallback to dummy key to avoid validation crash during app boot
        api_key = "DUMMY_KEY_FOR_BOOT"
    
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    # Check if the API Key actually exists at request time
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return jsonify({
            'error': 'OPENROUTER_API_KEY is missing. Please add it to your environment variables on Render.'
        }), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read the image file and convert to a base64 string
        image_bytes = image_file.read()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        mime_type = image_file.content_type or "image/jpeg"

        # Instantiate client lazily
        client = get_openai_client()

        # Call OpenRouter using the Nvidia Nemotron Omni free model
        response = client.chat.completions.create(
            model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all readable text from this image. Correct spelling errors and grammatical issues, structure it cleanly, and output ONLY the corrected final text. Do not include any introduction, comments, or summaries."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            extra_headers={
                "HTTP-Referer": SITE_URL,
                "X-Title": SITE_NAME,
            }
        )

        extracted_text = response.choices[0].message.content
        return jsonify({'text': extracted_text})

    except Exception as e:
        return jsonify({'error': f"API processing failed: {str(e)}"}), 500

@app.route('/save', methods=['POST'])
def save_text():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided to save'}), 400

    try:
        # Generate timestamped filename for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"saved_text_{timestamp}.txt"
        
        # Save to local directory on the server
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