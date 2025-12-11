"""
Flask Web Interface for Multimodal DeepFake Detection System
"""

import os
import shutil
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import tempfile
from predict import predict_video_with_audio_extraction

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle video or audio upload and analysis"""
    # Accept both video and audio keys
    file = request.files.get('video') or request.files.get('audio')
    if not file:
        return jsonify({'error': 'No file provided. Please upload a video or audio file.'}), 400

    # Check if file was selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    ext = file.filename.rsplit('.', 1)[-1].lower()
    is_video = ext in app.config['ALLOWED_EXTENSIONS']
    is_audio = ext in {'wav', 'mp3'}
    if not (is_video or is_audio):
        return jsonify({'error': 'Invalid file type. Allowed: mp4, avi, mov, mkv, wav, mp3'}), 400

    try:
        filename = secure_filename(file.filename)
        timestamp = str(int(os.times()[4] * 1000))
        unique_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)

        if is_video:
            # Process video and get prediction
            result = predict_video_with_audio_extraction(
                file_path,
                video_model_path='best_video_model.pth',
                audio_model_path='best_audio_model.h5',
                scaler_path='video_scaler.pkl'
            )

        else:
            from predict import predict_single_audio
            audio_result = predict_single_audio(file_path, model_path='best_audio_model.h5')
            if not audio_result or not audio_result.get('success'):
                result = {'error': audio_result.get('error', 'Could not process audio file.')}
            else:
                # Map backend keys to frontend expectations
                prob = audio_result.get('probability')
                label = 'DEEPFAKE' if prob > 0.5 else 'REAL'
                result = {
                    'video_prediction': None,
                    'audio_prediction': prob,
                    'final_prediction': label,
                    'confidence': audio_result.get('confidence', 0),
                    'video_label': None,
                    'audio_label': label
                }

        # Clean up uploaded file
        try:
            os.remove(file_path)
            if is_video:
                audio_path = file_path.rsplit('.', 1)[0] + '_audio.wav'
                if os.path.exists(audio_path):
                    os.remove(audio_path)
        except:
            pass

        if result is None:
            return jsonify({'error': 'Could not process file. Please ensure the file is valid.'}), 400

        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Check if models are loaded"""
    video_model_exists = os.path.exists('best_video_model.pth')
    audio_model_exists = os.path.exists('best_audio_model.h5')
    scaler_exists = os.path.exists('video_scaler.pkl')
    
    return jsonify({
        'status': 'ready' if (video_model_exists and audio_model_exists) else 'not_ready',
        'video_model': video_model_exists,
        'audio_model': audio_model_exists,
        'scaler': scaler_exists
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("DEEPFAKE DETECTOR WEB INTERFACE")
    print("="*60)
    print("\nStarting server...")
    print("Access the application at: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)