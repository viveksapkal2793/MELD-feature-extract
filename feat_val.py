# validate_embeddings.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

def comprehensive_validation(embedding_dir, sample_size=10):
    """Comprehensive validation of all embeddings"""
    
    # Load embeddings
    embedding_files = [f for f in os.listdir(embedding_dir) if f.endswith('.npy')]
    print(f"Found {len(embedding_files)} embedding files")
    
    if sample_size:
        embedding_files = embedding_files[:sample_size]
    
    embeddings = []
    names = []
    
    for file in embedding_files:
        emb = np.load(os.path.join(embedding_dir, file))
        embeddings.append(emb)
        names.append(file[:-4])  # Remove .npy
    
    embeddings = np.array(embeddings)
    print(f"Loaded embeddings shape: {embeddings.shape}")
    
    # 1. Basic statistics
    print("\n=== BASIC STATISTICS ===")
    print(f"Mean across all: {embeddings.mean():.6f}")
    print(f"Std across all: {embeddings.std():.6f}")
    print(f"Min value: {embeddings.min():.6f}")
    print(f"Max value: {embeddings.max():.6f}")
    
    # 2. Check for problematic embeddings
    print("\n=== PROBLEM DETECTION ===")
    nan_count = np.isnan(embeddings).sum()
    inf_count = np.isinf(embeddings).sum()
    
    # Fixed: Check for zero embeddings row by row
    zero_embeddings = 0
    for i, emb in enumerate(embeddings):
        if np.allclose(emb, 0):
            zero_embeddings += 1
            print(f"  Zero embedding found: {names[i]}")
    
    print(f"NaN values: {nan_count}")
    print(f"Inf values: {inf_count}")
    print(f"All-zero embeddings: {zero_embeddings}")
    
    # 3. Dimensionality check via PCA
    print("\n=== DIMENSIONALITY ANALYSIS ===")
    # Fix: Use min of samples, features, and desired components
    n_samples, n_features = embeddings.shape
    max_components = min(n_samples, n_features, 50)
    
    print(f"Dataset shape: {n_samples} samples × {n_features} features")
    print(f"Using {max_components} PCA components")
    
    if max_components > 1:
        pca = PCA(n_components=max_components)
        pca.fit(embeddings)
        
        # Explained variance ratio
        cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
        
        # Show relevant statistics based on available components
        if max_components >= 10:
            print(f"First 10 components explain {cumsum_variance[9]:.3f} of variance")
        else:
            print(f"First {max_components} components explain {cumsum_variance[-1]:.3f} of variance")
            
        if max_components >= 50:
            print(f"First 50 components explain {cumsum_variance[49]:.3f} of variance")
        
        print(f"All {max_components} components explain {cumsum_variance[-1]:.3f} of variance")
    else:
        print("Cannot perform PCA with current data size")
        cumsum_variance = np.array([1.0])
    
    # 4. Similarity analysis
    print("\n=== SIMILARITY ANALYSIS ===")
    similarity_matrix = cosine_similarity(embeddings)
    
    # Remove diagonal (self-similarity)
    mask = np.eye(similarity_matrix.shape[0], dtype=bool)
    similarities = similarity_matrix[~mask]
    
    print(f"Average cosine similarity: {similarities.mean():.3f}")
    print(f"Similarity std: {similarities.std():.3f}")
    print(f"Min similarity: {similarities.min():.3f}")
    print(f"Max similarity: {similarities.max():.3f}")
    
    # If similarities are too high, embeddings might be too similar
    if similarities.mean() > 0.9:
        print("WARNING: Very high similarities - embeddings might be too similar")
    elif similarities.mean() < 0.1:
        print("INFO: Low similarities - embeddings are quite diverse")
    else:
        print("INFO: Normal similarity range - embeddings look good")
    
    # 5. Additional quality checks
    print("\n=== QUALITY CHECKS ===")
    
    # Check L2 norms
    l2_norms = np.linalg.norm(embeddings, axis=1)
    print(f"L2 norm - Mean: {l2_norms.mean():.3f}, Std: {l2_norms.std():.3f}")
    print(f"L2 norm - Min: {l2_norms.min():.3f}, Max: {l2_norms.max():.3f}")
    
    # Check for near-constant embeddings
    embedding_stds = np.std(embeddings, axis=1)
    constant_embeddings = np.sum(embedding_stds < 1e-6)
    print(f"Near-constant embeddings (std < 1e-6): {constant_embeddings}")
    
    if constant_embeddings > 0:
        for i, std_val in enumerate(embedding_stds):
            if std_val < 1e-6:
                print(f"  Constant embedding: {names[i]} (std: {std_val:.8f})")
    
    # 6. Visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Histogram of embedding values
    plt.subplot(2, 3, 1)
    plt.hist(embeddings.flatten(), bins=50, alpha=0.7, edgecolor='black')
    plt.title('Distribution of Embedding Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: PCA explained variance (if available)
    plt.subplot(2, 3, 2)
    if max_components > 1:
        components_to_plot = min(20, max_components)
        plt.plot(range(1, components_to_plot + 1), cumsum_variance[:components_to_plot], 'o-')
        plt.title(f'Cumulative Explained Variance (First {components_to_plot} PCs)')
        plt.xlabel('Principal Component')
        plt.ylabel('Cumulative Variance Explained')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Not enough samples\nfor PCA analysis', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('PCA Analysis')
    
    # Plot 3: Similarity heatmap
    plt.subplot(2, 3, 3)
    sample_size_viz = min(10, len(embeddings))
    sample_sim = similarity_matrix[:sample_size_viz, :sample_size_viz]
    
    # Create shortened labels for better display
    short_names = [name[:8] + '...' if len(name) > 8 else name for name in names[:sample_size_viz]]
    
    sns.heatmap(sample_sim, annot=True, fmt='.2f', cmap='coolwarm', 
                xticklabels=short_names, yticklabels=short_names,
                cbar_kws={'label': 'Cosine Similarity'})
    plt.title('Cosine Similarity Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # Plot 4: L2 norms per embedding
    plt.subplot(2, 3, 4)
    bars = plt.bar(range(len(l2_norms)), l2_norms, alpha=0.7)
    plt.title('L2 Norms of Embeddings')
    plt.xlabel('Embedding Index')
    plt.ylabel('L2 Norm')
    plt.grid(True, alpha=0.3)
    
    # Add mean line
    plt.axhline(y=l2_norms.mean(), color='red', linestyle='--', 
                label=f'Mean: {l2_norms.mean():.2f}')
    plt.legend()
    
    # Plot 5: Standard deviation per embedding
    plt.subplot(2, 3, 5)
    bars = plt.bar(range(len(embedding_stds)), embedding_stds, alpha=0.7)
    plt.title('Standard Deviation per Embedding')
    plt.xlabel('Embedding Index')
    plt.ylabel('Std Dev')
    plt.grid(True, alpha=0.3)
    
    # Add mean line
    plt.axhline(y=embedding_stds.mean(), color='red', linestyle='--', 
                label=f'Mean: {embedding_stds.mean():.3f}')
    plt.legend()
    
    # Plot 6: Feature statistics across all embeddings
    plt.subplot(2, 3, 6)
    feature_means = np.mean(embeddings, axis=0)
    feature_stds = np.std(embeddings, axis=0)
    
    plt.plot(feature_means, alpha=0.7, label='Mean per feature')
    plt.fill_between(range(len(feature_means)), 
                     feature_means - feature_stds, 
                     feature_means + feature_stds, 
                     alpha=0.3, label='±1 std')
    plt.title('Feature Statistics Across Embeddings')
    plt.xlabel('Feature Index (first 100)' if len(feature_means) > 100 else 'Feature Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Limit x-axis for readability if too many features
    if len(feature_means) > 100:
        plt.xlim(0, 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(embedding_dir, 'validation_plots.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    # 7. Generate validation report
    report = {
        'total_embeddings': len(embeddings),
        'embedding_dim': embeddings.shape[1],
        'mean_value': float(embeddings.mean()),
        'std_value': float(embeddings.std()),
        'min_value': float(embeddings.min()),
        'max_value': float(embeddings.max()),
        'nan_count': int(nan_count),
        'inf_count': int(inf_count),
        'zero_embeddings': int(zero_embeddings),
        'constant_embeddings': int(constant_embeddings),
        'avg_cosine_similarity': float(similarities.mean()),
        'similarity_std': float(similarities.std()),
        'avg_l2_norm': float(l2_norms.mean()),
        'l2_norm_std': float(l2_norms.std()),
        'avg_embedding_std': float(embedding_stds.mean()),
        'pca_variance_explained': float(cumsum_variance[-1]) if max_components > 1 else 1.0,
        'is_valid': (nan_count == 0 and inf_count == 0 and zero_embeddings == 0 and
                    constant_embeddings == 0 and similarities.mean() < 0.95 and
                    l2_norms.mean() > 0.1 and embedding_stds.mean() > 0.01)
    }
    
    return report

def quick_sanity_check(embedding_path):
    """Quick check for a single embedding file"""
    emb = np.load(embedding_path)
    
    checks = {
        'correct_shape': len(emb.shape) == 1 and emb.shape[0] > 0,
        'no_nan': not np.isnan(emb).any(),
        'no_inf': not np.isinf(emb).any(),
        'not_all_zeros': not np.allclose(emb, 0),
        'reasonable_range': np.abs(emb).max() < 100,
        'non_zero_norm': np.linalg.norm(emb) > 1e-6,
        'has_variance': np.std(emb) > 1e-6
    }
    
    all_passed = all(checks.values())
    
    print(f"File: {os.path.basename(embedding_path)}")
    for check, passed in checks.items():
        print(f"  {check}: {'✓' if passed else '✗'}")
    print(f"  Overall: {'✓ VALID' if all_passed else '✗ INVALID'}")
    
    return all_passed

if __name__ == '__main__':
    embedding_dir = "D:/Acads/BTP/MELD/MELD.VideoFrames/dev_videomae_feat"
    report = comprehensive_validation(embedding_dir)
    print(f"\n=== FINAL REPORT ===")
    for key, value in report.items():
        print(f"{key}: {value}")
    
    # Also run quick check on first few files
    print(f"\n=== QUICK INDIVIDUAL CHECKS ===")
    embedding_files = [f for f in os.listdir(embedding_dir) if f.endswith('.npy')][:3]
    for file in embedding_files:
        quick_sanity_check(os.path.join(embedding_dir, file))
        print()