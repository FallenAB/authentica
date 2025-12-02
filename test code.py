"""
Prediction script for Multimodal DeepFake Detection System
"""

import os
import pickle
from deepfake_detector import MultimodalDeepFakeDetector

def predict_single_video(video_path, model_path='best_video_model.pth', 
                         scaler_path='video_scaler.pkl'):
    """Predict if a single video is deepfake"""
    
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


def predict_multimodal(video_path, audio_path, 
                       video_model='best_video_model.pth',
                       audio_model='best_audio_model.h5',
                       scaler_path='video_scaler.pkl'):
    """Predict using both video and audio"""
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
    
    if not os.path.exists(audio_path):
        print(f"❌ Audio not found: {audio_path}")
        return
    
    print("\n" + "="*60)
    print("MULTIMODAL DEEPFAKE DETECTION")
    print("="*60)
    print(f"Video: {video_path}")
    print(f"Audio: {audio_path}")
    
    # Initialize detector
    detector = MultimodalDeepFakeDetector(
        video_model_path=video_model,
        audio_model_path=audio_model
    )
    
    # Load scaler
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            detector.scaler = pickle.load(f)
    
    # Predict
    result = detector.predict_multimodal(video_path, audio_path)
    
    # Display results
    print("\n" + "-"*60)
    print("ANALYSIS RESULTS:")
    print("-"*60)
    
    if result['video_prediction'] is not None:
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
        if result['video_prediction'] and result['video_prediction'] > 0.5:
            print("   Reason: Video component detected as fake")
        if result['audio_prediction'] and result['audio_prediction'] > 0.5:
            print("   Reason: Audio component detected as fake")
    else:
        print("\n✅ This content appears to be REAL")
    
    print("="*60)


def batch_predict_videos(video_dir, output_file='predictions.csv'):
    """Predict all videos in a directory"""
    
    if not os.path.exists(video_dir):
        print(f"❌ Directory not found: {video_dir}")
        return
    
    print("\n" + "="*60)
    print("BATCH VIDEO PREDICTION")
    print("="*60)
    print(f"Directory: {video_dir}")
    
    # Initialize detector
    detector = MultimodalDeepFakeDetector(
        video_model_path='best_video_model.pth',
        audio_model_path=None
    )
    
    # Load scaler
    if os.path.exists('video_scaler.pkl'):
        with open('video_scaler.pkl', 'rb') as f:
            detector.scaler = pickle.load(f)
    
    # Process all videos
    results = []
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi'))]
    
    print(f"\nProcessing {len(video_files)} videos...")
    
    for i, video_file in enumerate(video_files):
        video_path = os.path.join(video_dir, video_file)
        print(f"\n[{i+1}/{len(video_files)}] {video_file}")
        
        prediction = detector.predict_video(video_path)
        
        if prediction is not None:
            label = 'DEEPFAKE' if prediction > 0.5 else 'REAL'
            print(f"  Result: {label} (confidence: {prediction:.4f})")
            results.append({
                'filename': video_file,
                'prediction': prediction,
                'label': label
            })
        else:
            print(f"  ⚠️  Could not process")
            results.append({
                'filename': video_file,
                'prediction': None,
                'label': 'ERROR'
            })
    
    # Save results
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
    print(f"\n✓ Results saved to: {output_file}")
    print("="*60)


def main():
    """Main prediction interface"""
    
    print("\n" + "="*60)
    print("MULTIMODAL DEEPFAKE DETECTOR - PREDICTION")
    print("="*60)
    
    print("\nUsage examples:")
    print("\n1. Predict single video:")
    print("   from predict import predict_single_video")
    print("   predict_single_video('path/to/video.mp4')")
    
    print("\n2. Predict single audio:")
    print("   from predict import predict_single_audio")
    print("   predict_single_audio('path/to/audio.wav')")
    
    print("\n3. Multimodal prediction:")
    print("   from predict import predict_multimodal")
    print("   predict_multimodal('video.mp4', 'audio.wav')")
    
    print("\n4. Batch prediction:")
    print("   from predict import batch_predict_videos")
    print("   batch_predict_videos('videos_folder/')")
    
    print("\n" + "="*60)
    
    # Example usage (uncomment to use)
    # predict_single_video('test_video.mp4')
    # predict_multimodal('test_video.mp4', 'test_audio.wav')


if __name__ == "__main__":
    main()