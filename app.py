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
def upload_video():
    """Handle video upload and analysis"""
    
    # Check if file was uploaded
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    
    # Check if file was selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check file extension
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: mp4, avi, mov, mkv'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = str(int(os.times()[4] * 1000))
        unique_filename = f"{timestamp}_{filename}"
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(video_path)
        
        # Process video and get prediction
        result = predict_video_with_audio_extraction(
            video_path,
            video_model_path='best_video_model.pth',
            audio_model_path='best_audio_model.h5',
            scaler_path='video_scaler.pkl'
        )
        
        # Clean up uploaded file
        try:
            os.remove(video_path)
            # Also remove extracted audio if it exists
            audio_path = video_path.rsplit('.', 1)[0] + '_audio.wav'
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except:
            pass
        
        if result is None:
            return jsonify({'error': 'Could not process video. Please ensure video contains faces.'}), 400
        
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