# @Description: Validate and analyze extracted HuBERT audio features

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import argparse
import json
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

def load_features(features_dir):
    """Load all .npy feature files from directory"""
    features = {}
    feature_files = [f for f in os.listdir(features_dir) if f.endswith('.npy')]
    
    print(f"Loading {len(feature_files)} feature files...")
    
    for i, file in enumerate(feature_files):
        if i % 1000 == 0:
            print(f"  Loaded {i}/{len(feature_files)} files...")
            
        video_name = os.path.splitext(file)[0]
        feature_path = os.path.join(features_dir, file)
        
        try:
            feature = np.load(feature_path)
            features[video_name] = feature
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    print(f"Successfully loaded {len(features)} features")
    return features

def analyze_feature_statistics(features):
    """Analyze basic statistics of features"""
    print("\n" + "="*50)
    print("FEATURE STATISTICS ANALYSIS")
    print("="*50)
    
    # Collect all features
    all_features = []
    shapes = []
    zero_count = 0
    
    for name, feature in features.items():
        shapes.append(feature.shape)
        all_features.append(feature.flatten())
        
        # Check for zero vectors
        if np.allclose(feature, 0):
            zero_count += 1
    
    # Shape analysis
    unique_shapes = list(set(shapes))
    print(f"📊 Unique feature shapes: {unique_shapes}")
    
    if len(unique_shapes) > 1:
        print("⚠️  WARNING: Multiple feature shapes detected!")
        for shape in unique_shapes:
            count = shapes.count(shape)
            print(f"   Shape {shape}: {count} files")
    
    # Zero vector analysis
    print(f"🕳️  Zero vectors: {zero_count}/{len(features)} ({100*zero_count/len(features):.1f}%)")
    
    if zero_count > len(features) * 0.1:  # More than 10% zeros
        print("⚠️  WARNING: High number of zero vectors detected!")
    
    # Statistical analysis
    if len(all_features) > 0:
        all_features = np.vstack(all_features)
        
        print(f"\n📈 Statistical Summary:")
        print(f"   Feature dimension: {all_features.shape[1]}")
        print(f"   Mean: {np.mean(all_features):.6f}")
        print(f"   Std: {np.std(all_features):.6f}")
        print(f"   Min: {np.min(all_features):.6f}")
        print(f"   Max: {np.max(all_features):.6f}")
        print(f"   Median: {np.median(all_features):.6f}")
        
        # Check for suspicious patterns
        if np.std(all_features) < 1e-6:
            print("⚠️  WARNING: Very low standard deviation - features might be constant!")
        
        if np.all(all_features >= 0):
            print("⚠️  WARNING: All values are non-negative - unusual for embeddings!")
        
        if np.all(all_features <= 0):
            print("⚠️  WARNING: All values are non-positive - unusual for embeddings!")
    
    return all_features

def analyze_feature_diversity(features, sample_size=100):
    """Analyze diversity and similarity between features"""
    print("\n" + "="*50)
    print("FEATURE DIVERSITY ANALYSIS")
    print("="*50)
    
    # Sample features for analysis
    feature_names = list(features.keys())
    if len(feature_names) > sample_size:
        sampled_names = np.random.choice(feature_names, sample_size, replace=False)
    else:
        sampled_names = feature_names
    
    sampled_features = [features[name] for name in sampled_names]
    
    # Calculate pairwise similarities
    similarities = []
    for i in range(len(sampled_features)):
        for j in range(i+1, len(sampled_features)):
            # Cosine similarity
            feat1 = sampled_features[i].flatten()
            feat2 = sampled_features[j].flatten()
            
            if np.linalg.norm(feat1) > 0 and np.linalg.norm(feat2) > 0:
                sim = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
                similarities.append(sim)
    
    if similarities:
        similarities = np.array(similarities)
        print(f"🔄 Pairwise Cosine Similarities (sample of {len(sampled_features)} features):")
        print(f"   Mean similarity: {np.mean(similarities):.4f}")
        print(f"   Std similarity: {np.std(similarities):.4f}")
        print(f"   Min similarity: {np.min(similarities):.4f}")
        print(f"   Max similarity: {np.max(similarities):.4f}")
        
        # Check for suspicious similarity patterns
        if np.mean(similarities) > 0.95:
            print("⚠️  WARNING: Very high average similarity - features might be too similar!")
        
        if np.std(similarities) < 0.01:
            print("⚠️  WARNING: Very low similarity variance - features might be identical!")
        
        return similarities
    else:
        print("❌ Could not compute similarities (zero vectors)")
        return np.array([])

def analyze_feature_distribution(all_features, save_dir=None):
    """Analyze and visualize feature distributions"""
    print("\n" + "="*50)
    print("FEATURE DISTRIBUTION ANALYSIS")
    print("="*50)
    
    if len(all_features) == 0:
        print("❌ No features to analyze")
        return
    
    # Sample dimensions for analysis
    n_dims = all_features.shape[1]
    sample_dims = min(10, n_dims)
    sampled_dim_indices = np.random.choice(n_dims, sample_dims, replace=False)
    
    print(f"📊 Analyzing distribution of {sample_dims} random dimensions (out of {n_dims})")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('HuBERT Feature Analysis', fontsize=16)
    
    # 1. Overall feature distribution
    axes[0, 0].hist(all_features.flatten(), bins=50, alpha=0.7, density=True)
    axes[0, 0].set_title('Overall Feature Value Distribution')
    axes[0, 0].set_xlabel('Feature Value')
    axes[0, 0].set_ylabel('Density')
    
    # 2. Per-dimension distributions (sample)
    for i, dim_idx in enumerate(sampled_dim_indices[:5]):
        axes[0, 1].hist(all_features[:, dim_idx], bins=30, alpha=0.5, 
                       label=f'Dim {dim_idx}', density=True)
    axes[0, 1].set_title('Sample Dimension Distributions')
    axes[0, 1].set_xlabel('Feature Value')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    
    # 3. Feature norms
    feature_norms = np.linalg.norm(all_features, axis=1)
    axes[1, 0].hist(feature_norms, bins=50, alpha=0.7)
    axes[1, 0].set_title('Feature Vector Norms')
    axes[1, 0].set_xlabel('L2 Norm')
    axes[1, 0].set_ylabel('Count')
    
    # 4. Dimension-wise statistics
    dim_means = np.mean(all_features, axis=0)
    dim_stds = np.std(all_features, axis=0)
    
    axes[1, 1].scatter(range(len(dim_means[:100])), dim_means[:100], alpha=0.6, s=10)
    axes[1, 1].set_title('Per-Dimension Means (first 100 dims)')
    axes[1, 1].set_xlabel('Dimension')
    axes[1, 1].set_ylabel('Mean Value')
    
    plt.tight_layout()
    
    if save_dir:
        plot_path = os.path.join(save_dir, 'feature_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Saved analysis plot to: {plot_path}")
    
    plt.show()
    
    # Statistical tests
    print(f"\n🧪 Statistical Tests:")
    
    # Normality test (sample)
    sample_features = all_features[:min(1000, len(all_features)), 0]  # First dimension, sample
    _, p_normal = stats.shapiro(sample_features)
    print(f"   Shapiro-Wilk normality test p-value: {p_normal:.6f}")
    
    # Feature norm analysis
    print(f"\n📏 Feature Norms:")
    print(f"   Mean norm: {np.mean(feature_norms):.4f}")
    print(f"   Std norm: {np.std(feature_norms):.4f}")
    print(f"   Min norm: {np.min(feature_norms):.4f}")
    print(f"   Max norm: {np.max(feature_norms):.4f}")
    
    # Check for zero norms
    zero_norms = np.sum(feature_norms < 1e-6)
    print(f"   Zero norm vectors: {zero_norms}/{len(feature_norms)}")

def validate_against_baseline(features, save_dir=None):
    """Validate features against expected HuBERT characteristics"""
    print("\n" + "="*50)
    print("HUBERT FEATURE VALIDATION")
    print("="*50)
    
    # Expected characteristics for HuBERT features
    expected_dim = 1024  # HuBERT-large
    
    validation_results = {
        'dimension_check': False,
        'non_constant_check': False,
        'reasonable_range_check': False,
        'diversity_check': False,
        'zero_ratio_check': False
    }
    
    # 1. Dimension check
    sample_feature = next(iter(features.values()))
    actual_dim = sample_feature.shape[-1] if sample_feature.ndim > 0 else 0
    
    if actual_dim == expected_dim:
        validation_results['dimension_check'] = True
        print(f"✅ Dimension check: {actual_dim} (expected: {expected_dim})")
    else:
        print(f"❌ Dimension check: {actual_dim} (expected: {expected_dim})")
    
    # 2. Non-constant check
    all_features = np.vstack([f.flatten() for f in features.values()])
    feature_std = np.std(all_features)
    
    if feature_std > 0.01:  # Reasonable threshold
        validation_results['non_constant_check'] = True
        print(f"✅ Non-constant check: std = {feature_std:.4f}")
    else:
        print(f"❌ Non-constant check: std = {feature_std:.4f} (too low)")
    
    # 3. Reasonable range check
    feature_min, feature_max = np.min(all_features), np.max(all_features)
    
    # HuBERT features typically range roughly in [-10, 10] but can vary
    if -50 < feature_min and feature_max < 50:
        validation_results['reasonable_range_check'] = True
        print(f"✅ Range check: [{feature_min:.3f}, {feature_max:.3f}]")
    else:
        print(f"⚠️  Range check: [{feature_min:.3f}, {feature_max:.3f}] (unusual range)")
    
    # 4. Diversity check
    sample_size = min(100, len(features))
    sample_features = list(features.values())[:sample_size]
    
    similarities = []
    for i in range(len(sample_features)):
        for j in range(i+1, len(sample_features)):
            feat1 = sample_features[i].flatten()
            feat2 = sample_features[j].flatten()
            
            if np.linalg.norm(feat1) > 0 and np.linalg.norm(feat2) > 0:
                sim = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
                similarities.append(sim)
    
    if similarities:
        mean_sim = np.mean(similarities)
        if mean_sim < 0.9:  # Features should not be too similar
            validation_results['diversity_check'] = True
            print(f"✅ Diversity check: mean similarity = {mean_sim:.4f}")
        else:
            print(f"❌ Diversity check: mean similarity = {mean_sim:.4f} (too high)")
    
    # 5. Zero ratio check
    zero_count = sum(1 for f in features.values() if np.allclose(f, 0))
    zero_ratio = zero_count / len(features)
    
    if zero_ratio < 0.1:  # Less than 10% zeros
        validation_results['zero_ratio_check'] = True
        print(f"✅ Zero ratio check: {zero_ratio:.2%}")
    else:
        print(f"❌ Zero ratio check: {zero_ratio:.2%} (too many zeros)")
    
    # Overall validation
    passed_checks = sum(validation_results.values())
    total_checks = len(validation_results)
    
    print(f"\n🎯 Overall Validation: {passed_checks}/{total_checks} checks passed")
    
    if passed_checks >= 4:
        print("✅ FEATURES APPEAR TO BE VALID!")
    elif passed_checks >= 3:
        print("⚠️  FEATURES ARE QUESTIONABLE - Some issues detected")
    else:
        print("❌ FEATURES APPEAR TO BE INVALID - Multiple issues detected")
    
    # Save validation report (FIX: Convert numpy types to native Python types)
    if save_dir:
        report = {
            'validation_results': validation_results,
            'checks_passed': int(passed_checks),  # Convert to int
            'total_checks': int(total_checks),    # Convert to int
            'feature_stats': {
                'total_features': int(len(features)),                    # Convert to int
                'feature_dimension': int(actual_dim),                   # Convert to int
                'zero_ratio': float(zero_ratio),                        # Convert to float
                'mean_similarity': float(np.mean(similarities)) if similarities else None,  # Convert to float
                'feature_std': float(feature_std),                      # Convert to float
                'feature_range': [float(feature_min), float(feature_max)]  # Convert to float
            }
        }
        
        report_path = os.path.join(save_dir, 'validation_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"📄 Saved validation report to: {report_path}")
    
    return validation_results

def main():
    parser = argparse.ArgumentParser(description='Validate HuBERT audio features')
    parser.add_argument('--features_dir', type=str, required=True,
                       help='Directory containing extracted .npy feature files')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save analysis results')
    parser.add_argument('--sample_size', type=int, default=100,
                       help='Sample size for diversity analysis')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.features_dir):
        print(f"❌ Features directory not found: {args.features_dir}")
        return
    
    if args.save_dir and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    print("🔍 HUBERT AUDIO FEATURES VALIDATION")
    print("="*60)
    print(f"Features directory: {args.features_dir}")
    print(f"Save directory: {args.save_dir}")
    
    # Load features
    features = load_features(args.features_dir)
    
    if len(features) == 0:
        print("❌ No features found!")
        return
    
    # Run analyses
    all_features = analyze_feature_statistics(features)
    similarities = analyze_feature_diversity(features, args.sample_size)
    analyze_feature_distribution(all_features, args.save_dir)
    validation_results = validate_against_baseline(features, args.save_dir)
    
    print("\n" + "="*60)
    print("🎉 ANALYSIS COMPLETE!")
    print("="*60)

if __name__ == '__main__':
    main()

# Usage examples:
# python validate_audio_features.py --features_dir "D:\Acads\BTP\MELD\MELD.VideoFrames\train_audio_feat"
# python validate_audio_features.py --features_dir "D:\Acads\BTP\MELD\MELD.VideoFrames\train_audio_feat" --save_dir "./validation_results"