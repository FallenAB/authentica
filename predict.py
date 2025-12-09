"""
Prediction script for Multimodal DeepFake Detection System
Updated with automatic audio extraction from video
"""

import os
import pickle
import subprocess
import tempfile
from deepfake_detector import MultimodalDeepFakeDetector

def extract_audio_from_video(video_path, audio_output_path=None):
    """
    Extract audio from video file using ffmpeg
    
    Args:
        video_path: Path to input video file
        audio_output_path: Path to save extracted audio (optional)
    
    Returns:
        Path to extracted audio file or None if failed
    """
    try:
        # Create output path if not provided
        if audio_output_path is None:
            base_path = video_path.rsplit('.', 1)[0]
            audio_output_path = base_path + '_audio.wav'
        
        # Check if ffmpeg is available
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except FileNotFoundError:
            print("❌ ffmpeg not found. Please install ffmpeg:")
            print("   Ubuntu/Debian: sudo apt-get install ffmpeg")
            print("   macOS: brew install ffmpeg")
            print("   Windows: Download from https://ffmpeg.org/download.html")
            return None
        
        # Extract audio using ffmpeg
        command = [
            'ffmpeg',
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # Audio codec
            '-ar', '44100',  # Sample rate
            '-ac', '2',  # Stereo
            '-y',  # Overwrite output file
            audio_output_path
        ]
        
        result = subprocess.run(
            command,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"❌ Failed to extract audio: {result.stderr}")
            return None
        
        if os.path.exists(audio_output_path):
            print(f"✅ Audio extracted to: {audio_output_path}")
            return audio_output_path
        else:
            print("❌ Audio extraction failed")
            return None
    
    except Exception as e:
        print(f"❌ Error extracting audio: {e}")
        return None


def predict_video_with_audio_extraction(video_path, 
                                         video_model_path='best_video_model.pth',
                                         audio_model_path='best_audio_model.h5',
                                         scaler_path='video_scaler.pkl',
                                         cleanup=True):
    """
    Predict if video is deepfake by analyzing both video and extracted audio
    
    Args:
        video_path: Path to video file
        video_model_path: Path to trained video model
        audio_model_path: Path to trained audio model
        scaler_path: Path to feature scaler
        cleanup: Whether to delete extracted audio after analysis
    
    Returns:
        dict: Prediction results
    """
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return None
    
    print("\n" + "="*60)
    print("MULTIMODAL DEEPFAKE DETECTION")
    print("="*60)
    print(f"Analyzing: {video_path}")
    
    # Extract audio from video
    print("\n📹 Extracting audio from video...")
    audio_path = extract_audio_from_video(video_path)
    
    if audio_path is None:
        print("⚠️  Could not extract audio, analyzing video only...")
        audio_path = None
    
    # Initialize detector
    detector = MultimodalDeepFakeDetector(
        video_model_path=video_model_path,
        audio_model_path=audio_model_path
    )
    
    # Load scaler
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            detector.scaler = pickle.load(f)
    else:
        print(f"⚠️  Scaler not found: {scaler_path}")
        print("Using default scaling (may affect accuracy)")
    
    # Analyze video
    print("\n📹 Analyzing video features...")
    video_pred = detector.predict_video(video_path)
    
    if video_pred is None:
        print("❌ Could not process video (no face detected)")
        if audio_path and cleanup:
            try:
                os.remove(audio_path)
            except:
                pass
        return None
    
    # Analyze audio if available
    audio_pred = None
    if audio_path and os.path.exists(audio_path):
        print("🔊 Analyzing audio features...")
        audio_pred = detector.predict_audio(audio_path)
        
        # Cleanup extracted audio if requested
        if cleanup:
            try:
                os.remove(audio_path)
                print(f"🗑️  Cleaned up temporary audio file")
            except Exception as e:
                print(f"⚠️  Could not delete audio file: {e}")
    
    # Determine final prediction
    threshold = 0.5
    
    if audio_pred is not None:
        # Multimodal: classify as deepfake if either component is fake
        is_deepfake = (video_pred > threshold) or (audio_pred > threshold)
        confidence = max(video_pred, audio_pred)
    else:
        # Video only
        is_deepfake = video_pred > threshold
        confidence = video_pred
    
    result = {
        'video_prediction': float(video_pred),
        'audio_prediction': float(audio_pred) if audio_pred is not None else None,
        'final_prediction': 'DEEPFAKE' if is_deepfake else 'REAL',
        'confidence': float(confidence),
        'video_label': 'DEEPFAKE' if video_pred > threshold else 'REAL',
        'audio_label': 'DEEPFAKE' if audio_pred and audio_pred > threshold else 'REAL'
    }
    
    # Display results
    print("\n" + "-"*60)
    print("ANALYSIS RESULTS:")
    print("-"*60)
    
    print(f"\n📹 VIDEO Analysis:")
    print(f"   Probability: {result['video_prediction']:.4f}")
    print(f"   Classification: {result['video_label']}")
    
    if result['audio_prediction'] is not None:
        print(f"\n🔊 AUDIO Analysis:")
        print(f"   Probability: {result['audio_prediction']:.4f}")
        print(f"   Classification: {result['audio_label']}")
    
    print(f"\n{'='*60}")
    print(f"FINAL VERDICT: {result['final_prediction']}")
    print(f"Overall Confidence: {result['confidence']:.4f}")
    print(f"{'='*60}")
    
    # Explanation
    if result['final_prediction'] == 'DEEPFAKE':
        print("\n⚠️  This content is classified as DEEPFAKE")
        if result['video_prediction'] > threshold:
            print("   Reason: Video component detected as fake")
        if result['audio_prediction'] and result['audio_prediction'] > threshold:
            print("   Reason: Audio component detected as fake")
    else:
        print("\n✅ This content appears to be REAL")
    
    print("="*60)
    
    return result


def predict_single_video(video_path, model_path='best_video_model.pth', 
                         scaler_path='video_scaler.pkl'):
    """Predict if a single video is deepfake (video analysis only)"""
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
    
    print("\n" + "="*60)
    print("VIDEO DEEPFAKE DETECTION")
    print("="*60)
    print(f"Analyzing: {video_path}")
    
    # Initialize detector
    detector = MultimodalDeepFakeDetector(
        video_model_path=model_path,
        audio_model_path=None  # Only video
    )
    
    # Load scaler
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            detector.scaler = pickle.load(f)
    else:
        print(f"⚠️  Scaler not found: {scaler_path}")
        print("Using default scaling (may affect accuracy)")
    
    # Predict
    prediction = detector.predict_video(video_path)
    
    if prediction is None:
        print("❌ Could not process video (no face detected)")
        return
    
    # Display results
    print("\n" + "-"*60)
    print("RESULTS:")
    print("-"*60)
    print(f"Probability: {prediction:.4f}")
    print(f"Prediction: {'🚨 DEEPFAKE' if prediction > 0.5 else '✅ REAL'}")
    print(f"Confidence: {abs(prediction - 0.5) * 200:.1f}%")
    print("="*60)


def predict_single_audio(audio_path, model_path='best_audio_model.h5'):
    """Predict if a single audio file is deepfake"""
    
    if not os.path.exists(audio_path):
        print(f"❌ Audio not found: {audio_path}")
        return
    
    print("\n" + "="*60)
    print("AUDIO DEEPFAKE DETECTION")
    print("="*60)
    print(f"Analyzing: {audio_path}")
    
    # Initialize detector
    detector = MultimodalDeepFakeDetector(
        video_model_path=None,  # Only audio
        audio_model_path=model_path
    )
    
    # Predict
    prediction = detector.predict_audio(audio_path)
    
    if prediction is None:
        print("❌ Could not process audio")
        return
    
    # Display results
    print("\n" + "-"*60)
    print("RESULTS:")
    print("-"*60)
    print(f"Probability: {prediction:.4f}")
    print(f"Prediction: {'🚨 DEEPFAKE' if prediction > 0.5 else '✅ REAL'}")
    print(f"Confidence: {abs(prediction - 0.5) * 200:.1f}%")
    print("="*60)


def batch_predict_videos(video_dir, output_file='predictions.csv'):
    """Predict all videos in a directory with audio extraction"""
    
    if not os.path.exists(video_dir):
        print(f"❌ Directory not found: {video_dir}")
        return
    
    print("\n" + "="*60)
    print("BATCH VIDEO PREDICTION WITH AUDIO EXTRACTION")
    print("="*60)
    print(f"Directory: {video_dir}")
    
    # Process all videos
    results = []
    video_files = [f for f in os.listdir(video_dir) 
                   if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    print(f"\nProcessing {len(video_files)} videos...")
    
    for i, video_file in enumerate(video_files):
        video_path = os.path.join(video_dir, video_file)
        print(f"\n[{i+1}/{len(video_files)}] {video_file}")
        
        result = predict_video_with_audio_extraction(
            video_path,
            cleanup=True
        )
        
        if result is not None:
            print(f"  Result: {result['final_prediction']} "
                  f"(confidence: {result['confidence']:.4f})")
            
            results.append({
                'filename': video_file,
                'final_prediction': result['final_prediction'],
                'confidence': result['confidence'],
                'video_prediction': result['video_prediction'],
                'audio_prediction': result['audio_prediction'],
                'video_label': result['video_label'],
                'audio_label': result['audio_label']
            })
        else:
            print(f"  ⚠️  Could not process")
            results.append({
                'filename': video_file,
                'final_prediction': 'ERROR',
                'confidence': None,
                'video_prediction': None,
                'audio_prediction': None,
                'video_label': 'ERROR',
                'audio_label': 'ERROR'
            })
    
    # Save results
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Results saved to: {output_file}")
    print("="*60)


def main():
    """Main prediction interface"""
    
    print("\n" + "="*60)
    print("MULTIMODAL DEEPFAKE DETECTOR - PREDICTION")
    print("="*60)
    
    print("\nUsage examples:")
    print("\n1. Analyze video (automatic audio extraction):")
    print("   from predict import predict_video_with_audio_extraction")
    print("   predict_video_with_audio_extraction('path/to/video.mp4')")
    
    print("\n2. Predict single video only:")
    print("   from predict import predict_single_video")
    print("   predict_single_video('path/to/video.mp4')")
    
    print("\n3. Predict single audio:")
    print("   from predict import predict_single_audio")
    print("   predict_single_audio('path/to/audio.wav')")
    
    print("\n4. Batch prediction with audio extraction:")
    print("   from predict import batch_predict_videos")
    print("   batch_predict_videos('videos_folder/')")
    
    print("\n" + "="*60)
    predict_video_with_audio_extraction('H:/Python/test_video.mp4')
    # Example usage (uncomment to use)
    # predict_video_with_audio_extraction('test_video.mp4')


if __name__ == "__main__":
    main()