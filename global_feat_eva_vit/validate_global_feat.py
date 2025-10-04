#!/usr/bin/env python3
"""
Feature validation script for EVA-ViT extracted features.
Performs comprehensive checks to ensure feature quality and integrity.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from tqdm import tqdm
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class FeatureValidator:
    def __init__(self, features_dir, sample_images_dir=None):
        """
        Initialize feature validator
        
        Args:
            features_dir: Directory containing .npy feature files
            sample_images_dir: Directory containing sample images for visual validation
        """
        self.features_dir = Path(features_dir)
        self.sample_images_dir = Path(sample_images_dir) if sample_images_dir else None
        self.feature_files = list(self.features_dir.glob("*.npy"))
        
        print(f"Found {len(self.feature_files)} feature files")
        
        # Expected feature dimensions for EVA-ViT-g
        self.expected_shape = (1025, 1408)  # [CLS + patches, feature_dim]
        
    def validate_basic_properties(self):
        """Check basic properties of all feature files"""
        print("\n" + "="*60)
        print("BASIC PROPERTY VALIDATION")
        print("="*60)
        
        valid_files = 0
        invalid_files = []
        file_sizes = []
        
        for feature_file in tqdm(self.feature_files, desc="Validating basic properties"):
            try:
                features = np.load(feature_file)
                
                # Check shape
                if features.shape != self.expected_shape:
                    invalid_files.append({
                        'file': feature_file.name,
                        'issue': f'Wrong shape: {features.shape}, expected: {self.expected_shape}'
                    })
                    continue
                
                # Check for NaN or Inf values
                if np.isnan(features).any():
                    invalid_files.append({
                        'file': feature_file.name,
                        'issue': 'Contains NaN values'
                    })
                    continue
                
                if np.isinf(features).any():
                    invalid_files.append({
                        'file': feature_file.name,
                        'issue': 'Contains Inf values'
                    })
                    continue
                
                # Check data type
                if features.dtype not in [np.float32, np.float64, np.float16]:
                    invalid_files.append({
                        'file': feature_file.name,
                        'issue': f'Unexpected dtype: {features.dtype}'
                    })
                    continue
                
                # File is valid
                valid_files += 1
                file_sizes.append(features.nbytes / (1024*1024))  # Size in MB
                
            except Exception as e:
                invalid_files.append({
                    'file': feature_file.name,
                    'issue': f'Loading error: {str(e)}'
                })
        
        # Print results
        print(f"✓ Valid files: {valid_files}/{len(self.feature_files)}")
        print(f"✗ Invalid files: {len(invalid_files)}")
        
        if invalid_files:
            print("\nInvalid files:")
            for issue in invalid_files[:10]:  # Show first 10 issues
                print(f"  - {issue['file']}: {issue['issue']}")
            if len(invalid_files) > 10:
                print(f"  ... and {len(invalid_files) - 10} more")
        
        if file_sizes:
            print(f"\nFile sizes: {np.mean(file_sizes):.2f} ± {np.std(file_sizes):.2f} MB")
        
        return valid_files == len(self.feature_files)
    
    def analyze_feature_statistics(self, n_samples=50):
        """Analyze statistical properties of features"""
        print("\n" + "="*60)
        print("STATISTICAL ANALYSIS")
        print("="*60)
        
        # Load sample features
        sample_files = np.random.choice(self.feature_files, 
                                      min(n_samples, len(self.feature_files)), 
                                      replace=False)
        
        all_features = []
        cls_tokens = []
        patch_features = []
        
        print(f"Loading {len(sample_files)} sample files...")
        for feature_file in tqdm(sample_files):
            try:
                features = np.load(feature_file)
                all_features.append(features.flatten())
                cls_tokens.append(features[0])  # CLS token
                patch_features.append(features[1:].flatten())  # Patch tokens
            except Exception as e:
                print(f"Error loading {feature_file}: {e}")
        
        if not all_features:
            print("No valid features loaded for analysis!")
            return False
        
        all_features = np.array(all_features)
        cls_tokens = np.array(cls_tokens)
        patch_features = np.array(patch_features)
        
        # Overall statistics
        print(f"\nOverall Statistics:")
        print(f"  Mean: {np.mean(all_features):.6f}")
        print(f"  Std:  {np.std(all_features):.6f}")
        print(f"  Min:  {np.min(all_features):.6f}")
        print(f"  Max:  {np.max(all_features):.6f}")
        
        # CLS token statistics
        print(f"\nCLS Token Statistics:")
        print(f"  Mean: {np.mean(cls_tokens):.6f}")
        print(f"  Std:  {np.std(cls_tokens):.6f}")
        
        # Patch token statistics
        print(f"\nPatch Token Statistics:")
        print(f"  Mean: {np.mean(patch_features):.6f}")
        print(f"  Std:  {np.std(patch_features):.6f}")
        
        # Check for reasonable ranges (EVA-ViT typically produces values in [-10, 10] range)
        reasonable_range = (-20, 20)
        out_of_range = (all_features < reasonable_range[0]) | (all_features > reasonable_range[1])
        out_of_range_percent = np.sum(out_of_range) / all_features.size * 100
        
        print(f"\nRange Analysis:")
        print(f"  Values outside [{reasonable_range[0]}, {reasonable_range[1]}]: {out_of_range_percent:.2f}%")
        
        if out_of_range_percent > 5:
            print("  ⚠️  Warning: High percentage of values outside expected range!")
        else:
            print("  ✓ Values within expected range")
        
        return True
    
    def check_feature_diversity(self, n_samples=20):
        """Check that features are diverse and not all identical"""
        print("\n" + "="*60)
        print("FEATURE DIVERSITY CHECK")
        print("="*60)
        
        sample_files = np.random.choice(self.feature_files, 
                                      min(n_samples, len(self.feature_files)), 
                                      replace=False)
        
        features_list = []
        for feature_file in sample_files:
            try:
                features = np.load(feature_file)
                features_list.append(features.flatten())
            except:
                continue
        
        if len(features_list) < 2:
            print("Not enough valid features for diversity check")
            return False
        
        features_matrix = np.array(features_list)
        
        # Calculate pairwise correlations
        correlations = []
        for i in range(len(features_matrix)):
            for j in range(i+1, len(features_matrix)):
                corr = np.corrcoef(features_matrix[i], features_matrix[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        avg_correlation = np.mean(correlations) if correlations else 0
        max_correlation = np.max(correlations) if correlations else 0
        
        print(f"Pairwise correlations between features:")
        print(f"  Average absolute correlation: {avg_correlation:.4f}")
        print(f"  Maximum absolute correlation: {max_correlation:.4f}")
        
        # Check variance across features
        feature_variances = np.var(features_matrix, axis=0)
        low_variance_dims = np.sum(feature_variances < 1e-6)
        
        print(f"\nFeature variance analysis:")
        print(f"  Dimensions with very low variance: {low_variance_dims}/{features_matrix.shape[1]}")
        print(f"  Mean variance: {np.mean(feature_variances):.6f}")
        
        # Assessment
        diverse = True
        if avg_correlation > 0.9:
            print("  ⚠️  Warning: Features are highly correlated - may be identical!")
            diverse = False
        elif avg_correlation > 0.7:
            print("  ⚠️  Warning: Features are moderately correlated")
        else:
            print("  ✓ Features show good diversity")
        
        if low_variance_dims > features_matrix.shape[1] * 0.5:
            print("  ⚠️  Warning: Many dimensions have very low variance")
            diverse = False
        
        return diverse
    
    def create_validation_report(self, output_dir="validation_output"):
        """Create comprehensive validation report with plots"""
        print("\n" + "="*60)
        print("CREATING VALIDATION REPORT")
        print("="*60)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load sample features for plotting
        n_plot_samples = min(10, len(self.feature_files))
        sample_files = np.random.choice(self.feature_files, n_plot_samples, replace=False)
        
        sample_features = []
        sample_names = []
        
        for feature_file in sample_files:
            try:
                features = np.load(feature_file)
                sample_features.append(features)
                sample_names.append(feature_file.stem)
            except:
                continue
        
        if not sample_features:
            print("No valid features for plotting!")
            return
        
        # Plot 1: Feature distribution histograms
        plt.figure(figsize=(15, 10))
        
        for i, (features, name) in enumerate(zip(sample_features[:4], sample_names[:4])):
            plt.subplot(2, 2, i+1)
            plt.hist(features.flatten(), bins=50, alpha=0.7, density=True)
            plt.title(f'Feature Distribution: {name}')
            plt.xlabel('Feature Value')
            plt.ylabel('Density')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_distributions.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot 2: CLS vs Patch token comparison
        plt.figure(figsize=(12, 8))
        
        cls_means = []
        patch_means = []
        
        for features in sample_features:
            cls_means.append(np.mean(features[0]))  # CLS token mean
            patch_means.append(np.mean(features[1:]))  # Patch tokens mean
        
        plt.subplot(1, 2, 1)
        plt.scatter(cls_means, patch_means, alpha=0.7)
        plt.xlabel('CLS Token Mean')
        plt.ylabel('Patch Tokens Mean')
        plt.title('CLS vs Patch Token Comparison')
        plt.grid(True, alpha=0.3)
        
        # Add correlation
        if len(cls_means) > 1:
            corr = np.corrcoef(cls_means, patch_means)[0, 1]
            plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=plt.gca().transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 3: Feature magnitude heatmap
        plt.subplot(1, 2, 2)
        if sample_features:
            # Take first feature as example
            feature_2d = sample_features[0].reshape(32, -1)[:16, :44]  # Subsample for visualization
            sns.heatmap(feature_2d, cmap='RdBu_r', center=0, cbar=True)
            plt.title('Feature Magnitude Heatmap (Sample)')
            plt.xlabel('Feature Dimension (subsampled)')
            plt.ylabel('Token Position (subsampled)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create summary statistics file
        stats_data = []
        for feature_file in tqdm(self.feature_files[:100], desc="Computing statistics"):  # Limit for speed
            try:
                features = np.load(feature_file)
                stats_data.append({
                    'filename': feature_file.name,
                    'mean': np.mean(features),
                    'std': np.std(features),
                    'min': np.min(features),
                    'max': np.max(features),
                    'cls_token_mean': np.mean(features[0]),
                    'patch_tokens_mean': np.mean(features[1:])
                })
            except:
                continue
        
        if stats_data:
            df = pd.DataFrame(stats_data)
            df.to_csv(os.path.join(output_dir, 'feature_statistics.csv'), index=False)
            
            print(f"\nSummary statistics saved to: {os.path.join(output_dir, 'feature_statistics.csv')}")
            print(f"Plots saved to: {output_dir}/")
    
    def run_full_validation(self):
        """Run complete validation pipeline"""
        print("EVA-ViT FEATURE VALIDATION")
        print("="*60)
        print(f"Features directory: {self.features_dir}")
        print(f"Expected shape: {self.expected_shape}")
        print(f"Total files: {len(self.feature_files)}")
        
        # Run all validation checks
        basic_valid = self.validate_basic_properties()
        stats_valid = self.analyze_feature_statistics()
        diversity_valid = self.check_feature_diversity()
        
        # Create report
        self.create_validation_report()
        
        # Final assessment
        print("\n" + "="*60)
        print("FINAL VALIDATION RESULTS")
        print("="*60)
        
        print(f"✓ Basic Properties: {'PASS' if basic_valid else 'FAIL'}")
        print(f"✓ Statistical Analysis: {'PASS' if stats_valid else 'FAIL'}")
        print(f"✓ Feature Diversity: {'PASS' if diversity_valid else 'FAIL'}")
        
        overall_valid = basic_valid and stats_valid and diversity_valid
        print(f"\n🎯 Overall Validation: {'✅ PASS' if overall_valid else '❌ FAIL'}")
        
        if overall_valid:
            print("\n✅ Your features look good and ready to use!")
        else:
            print("\n⚠️  Some issues detected. Check the detailed output above.")
        
        return overall_valid

def main():
    parser = argparse.ArgumentParser(description="Validate extracted EVA-ViT features")
    parser.add_argument("--features_dir", required=True, help="Directory containing .npy feature files")
    parser.add_argument("--sample_images_dir", help="Directory containing sample images (optional)")
    parser.add_argument("--output_dir", default="validation_output", help="Output directory for validation report")
    parser.add_argument("--n_samples", default=50, type=int, help="Number of samples for statistical analysis")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.features_dir):
        print(f"Error: Features directory does not exist: {args.features_dir}")
        return
    
    # Initialize validator
    validator = FeatureValidator(
        features_dir=args.features_dir,
        sample_images_dir=args.sample_images_dir
    )
    
    # Run validation
    validator.run_full_validation()

if __name__ == "__main__":
    main()