import os
import argparse
import warnings
from transformers import HubertModel, Wav2Vec2FeatureExtractor

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

def download_hubert_model(model_name, local_dir):
    """Download HuBERT model and save it locally"""
    
    print(f"Downloading {model_name}...")
    print(f"Saving to: {local_dir}")
    
    # Create directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    try:
        # Download model and feature extractor
        print("Downloading HuBERT model...")
        model = HubertModel.from_pretrained(model_name)
        
        print("Downloading feature extractor...")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        
        # Save to local directory
        print("Saving model locally...")
        model.save_pretrained(local_dir)
        feature_extractor.save_pretrained(local_dir)
        
        # Verify files were saved
        saved_files = os.listdir(local_dir)
        if len(saved_files) > 0:
            print(f"✅ Model successfully downloaded and saved to: {local_dir}")
            print(f"📁 Downloaded files: {saved_files}")
            return True
        else:
            print("❌ No files were saved")
            return False
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download HuBERT model locally')
    parser.add_argument('--model_name', type=str, default='facebook/hubert-large-ls960-ft',
                       help='HuBERT model name from Hugging Face')
    parser.add_argument('--local_dir', type=str, default='./models/hubert-large-english',
                       help='Local directory to save the model')
    
    args = parser.parse_args()
    
    success = download_hubert_model(args.model_name, args.local_dir)
    
    if success:
        print("\n" + "="*50)
        print("DOWNLOAD COMPLETED!")
        print("="*50)
        print(f"Model saved in: {os.path.abspath(args.local_dir)}")
        print("\nYou can now use the model with:")
        print(f"--hubert_model {args.local_dir}")
        
        # Show directory contents and sizes
        local_dir = args.local_dir
        print(f"\nDirectory contents:")
        total_size = 0
        for file in os.listdir(local_dir):
            file_path = os.path.join(local_dir, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                total_size += size
                size_mb = size / (1024 * 1024)
                print(f"  📄 {file}: {size_mb:.1f} MB")
        print(f"📊 Total size: {total_size / (1024 * 1024):.1f} MB")
    else:
        print("Download failed. Please check your internet connection and try again.")