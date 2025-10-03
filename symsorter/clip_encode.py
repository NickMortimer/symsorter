import typer
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from doit.doit_cmd import DoitMain
from doit.cmd_base import ModuleTaskLoader
from doit.tools import run_once
import pandas as pd
from .config import cfg
from doit.tools import check_timestamp_unchanged
from transformers import Dinov2Model, AutoImageProcessor
import timm

# Try to import CLIP and set availability flag
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

app = typer.Typer()

DOIT_CONFIG = {'check_file_uptodate': 'timestamp'}

def load_clip_model(device='cuda'):
    """Load CLIP model and preprocessing function."""
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess

def load_existing_embeddings(npz_path):
    """Helper function to load existing embeddings from npz file"""
    existing_embeddings = {}
    existing_dimensions = {}
    metadata = {}
    if npz_path.exists():
        try:
            data = np.load(npz_path, allow_pickle=True)
            if 'embeddings' in data and 'filenames' in data:
                # Ensure all filenames are strings to avoid Path object issues
                filenames = [str(f) for f in data['filenames']]
                existing_embeddings = dict(zip(filenames, data['embeddings']))
                
                # Load dimensions if available (for backward compatibility)
                if 'dimensions' in data:
                    existing_dimensions = dict(zip(filenames, data['dimensions']))
                
                # Load metadata for concatenation compatibility
                if 'crop_size' in data:
                    # New format: array of crop sizes (should all be the same)
                    metadata['crop_size'] = int(data['crop_size'][0])
                
                if 'model_type' in data:
                    # New format: array of model types (should all be the same)
                    metadata['model_type'] = str(data['model_type'][0])
                
                print(f"Loaded {len(existing_embeddings)} existing embeddings from {npz_path}")
                if metadata:
                    print(f"  Metadata: {metadata}")
        except Exception as e:
            print(f"Error loading existing embeddings: {e}")
    return existing_embeddings, existing_dimensions, metadata

def save_folder_embeddings(npz_path, embeddings_dict, dimensions_dict=None, crop_size=0, model_type='clip'):
    """Helper function to save embeddings dictionary to npz file"""
    if not embeddings_dict:
        print(f"No embeddings to save for {npz_path}")
        return
    
    # Convert all keys to strings to avoid pickle issues with Path objects
    filenames = [str(key) for key in embeddings_dict.keys()]
    embeddings = np.array(list(embeddings_dict.values()))
    
    # Create dimensions array in same order as filenames if provided
    dimensions = None
    if dimensions_dict:
        original_keys = list(embeddings_dict.keys())
        dimensions = np.array([dimensions_dict.get(key, (0, 0)) for key in original_keys])
    
    # Prepare save data - make crop_size and model_type arrays matching filenames length
    crop_sizes = np.full(len(filenames), crop_size, dtype=np.int32)
    model_types = np.full(len(filenames), model_type, dtype='<U20')  # String array with max 20 chars
    
    save_data = {
        'embeddings': embeddings,
        'filenames': filenames,
        'dimensions': dimensions,
        'crop_size': crop_sizes,
        'model_type': model_types
    }
    
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **save_data)
    
    dims_info = f" with dimensions" if dimensions_dict else ""
    print(f"Saved {len(embeddings)} embeddings{dims_info} to {npz_path}")

def load_dinov3_model(model_name='dinov2-base', device='cuda'):
    """Load DINOv3 model and preprocessing function."""
    # You can use 'dinov2-small', 'dinov2-base', 'dinov2-large', 'dinov2-giant'
    processor = AutoImageProcessor.from_pretrained(f'facebook/{model_name}', use_fast=True)
    model = Dinov2Model.from_pretrained(f'facebook/{model_name}')
    model = model.to(device)
    model.eval()
    return model, processor

# Alternative: Using timm (often faster)
def load_dinov3_timm(model_name='vit_base_patch14_dinov2.lvd142m', device='cuda'):
    """Load DINOv3 via timm (alternative approach)."""
    model = timm.create_model(model_name, pretrained=True, num_classes=0)  # num_classes=0 for feature extraction
    model = model.to(device)
    model.eval()
    
    # Get the default data config for preprocessing
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    
    return model, transforms

def task_encode_folders():
    """Doit task generator for encoding image folders."""
    
    def image_to_pickle_name(image_path, crop_size, model_type='clip'):
        return image_path.parent / f'encoding_{model_type}_{crop_size:03d}.npz'

    def encode_folder_action(dependencies, targets):
        device = "cuda" if torch.cuda.is_available() else "cpu"
      
        # Load appropriate model
        print(f"📦 Loading {model_type} model...")
        if model_type == 'clip':
            model, preprocess_fn = clip.load("ViT-B/32", device=device)
        elif model_type == 'dinov3':
            model, preprocess_fn = load_dinov3_model('dinov2-base', device)
        elif model_type == 'dinov3_timm':
            model, preprocess_fn = load_dinov3_timm('vit_base_patch14_dinov2.lvd142m', device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        print(f"✅ Model loaded successfully on {device}")
        dependencies.sort()

        def load_and_crop(image):
            data = Image.open(image).convert("RGB")
            original_size = data.size  # Capture original dimensions
            if crop_size > 0:
                w, h = data.size
                side = min(w, h, crop_size)
                left = max((w - side) // 2, 0)
                top = max((h - side) // 2, 0)
                right = left + side
                bottom = top + side
                data = data.crop((left, top, right, bottom))
                if side != crop_size:
                    data = data.resize((crop_size, crop_size))
            return data, original_size
        
        def encode_batch_clip(image_paths):
            """Encode batch using CLIP."""
            batch_data = [load_and_crop(p) for p in image_paths]
            raw_images = [data[0] for data in batch_data]
            dimensions = [data[1] for data in batch_data]
            
            images = [preprocess_fn(img) for img in raw_images]
            batch = torch.stack(images).to(device)

            with torch.no_grad():
                feats = model.encode_image(batch)
                feats /= feats.norm(dim=-1, keepdim=True)
            return feats.cpu().numpy(), dimensions

        def encode_batch_dinov3(image_paths):
            """Encode batch using DINOv3 (transformers)."""
            batch_data = [load_and_crop(p) for p in image_paths]
            raw_images = [data[0] for data in batch_data]
            dimensions = [data[1] for data in batch_data]
            
            inputs = preprocess_fn(images=raw_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                # Use CLS token representation
                feats = outputs.last_hidden_state[:, 0]  # Shape: [batch_size, hidden_size]
                feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.cpu().numpy(), dimensions

        def encode_batch_dinov3_timm(image_paths):
            """Encode batch using DINOv3 (timm)."""
            batch_data = [load_and_crop(p) for p in image_paths]
            raw_images = [data[0] for data in batch_data]
            dimensions = [data[1] for data in batch_data]
            
            images = [preprocess_fn(img) for img in raw_images]
            batch = torch.stack(images).to(device)

            with torch.no_grad():
                feats = model(batch)  # Already returns features
                feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.cpu().numpy(), dimensions

        # Choose encoding function based on model type
        if model_type == 'clip':
            encode_batch = encode_batch_clip
        elif model_type == 'dinov3':
            encode_batch = encode_batch_dinov3
        elif model_type == 'dinov3_timm':
            encode_batch = encode_batch_dinov3_timm

        # Prepare the batches
        def chunked(dependencies, size):
            for i in range(0, len(dependencies), size):
                yield dependencies[i:i + size]
        
        # Process all images in large batches with progress bar
        all_embeddings = []
        all_dimensions = []
        batch_size = cfg.get('encode_batch_size')
        num_batches = (len(dependencies) + batch_size - 1) // batch_size
        for batch in tqdm(chunked(dependencies, batch_size), total=num_batches, desc=f"Encoding batches ({model_type})"):
            batch_embeddings, batch_dimensions = encode_batch(batch)
            all_embeddings.extend(batch_embeddings)
            all_dimensions.extend(batch_dimensions)
        
        # Now organize by folder and save to npz files
        current_folder = None
        current_npz_path = None
        existing_embeddings = {}
        existing_dimensions = {}
        folder_embeddings = []
        folder_filenames = []

        for image_path, embedding, dimensions in tqdm(zip(dependencies, all_embeddings, all_dimensions), total=len(dependencies), desc="Saving embeddings"):
            image_path = Path(image_path)
            npz_path = image_to_pickle_name(image_path, crop_size, model_type)
            relative_name = image_path.relative_to(npz_path.parent)
            # If we've moved to a new folder, save the previous one
            if npz_path != current_npz_path:
                if current_npz_path is not None:
                    # Save previous folder's data
                    save_folder_embeddings(current_npz_path, existing_embeddings, existing_dimensions, crop_size, model_type)

                # Load existing embeddings for new folder
                current_npz_path = npz_path
                existing_embeddings, existing_dimensions, existing_metadata = load_existing_embeddings(current_npz_path)
                
                # Check compatibility if existing file has metadata
                if existing_metadata:
                    if existing_metadata.get('crop_size') != crop_size:
                        print(f"Warning: crop_size mismatch in {npz_path}: existing={existing_metadata.get('crop_size')}, current={crop_size}")
                    if existing_metadata.get('model_type') != model_type:
                        print(f"Warning: model_type mismatch in {npz_path}: existing={existing_metadata.get('model_type')}, current={model_type}")
            
            # Add current image's embedding and dimensions if not already in existing
            # Convert to string to avoid pickle issues with Path objects
            existing_embeddings[str(relative_name)] = embedding
            existing_dimensions[str(relative_name)] = dimensions
        # Save the last folder
        if current_npz_path is not None:
            save_folder_embeddings(current_npz_path, existing_embeddings, existing_dimensions, crop_size, model_type)
    crop_size = cfg.get('encode_crop_size', 0)
    model_type = cfg.get('encode_model_type', 'clip')
    input_dir = cfg.get_url('encode_input_dir')
    pattern = cfg.get('encode_pattern')
    
    # get all jpgs in subfolders
    all_files = list(input_dir.rglob(f'**/{pattern.lower()}')) + list(input_dir.rglob(f'**/{pattern.upper()}'))
    
    # Group files by their npz path first
    files_by_npz = {}
    for file_path in all_files:
        npz_path = image_to_pickle_name(file_path, crop_size, model_type)
        if npz_path not in files_by_npz:
            files_by_npz[npz_path] = []
        files_by_npz[npz_path].append(file_path)
    
    # Filter to only include files that need processing
    files_to_process = []
    for npz_path, file_paths in files_by_npz.items():
        # Load existing embeddings once per npz file
        existing_filenames = set()
        if npz_path.exists():
            try:
                data = np.load(npz_path, allow_pickle=True)
                if 'embeddings' in data and 'filenames' in data:
                    existing_filenames = set(str(f) for f in data['filenames'])
            except Exception:
                pass  # If we can't load, assume all files need processing
        
        # Check each file in this group
        for file_path in file_paths:
            relative_name = file_path.relative_to(npz_path.parent)
            # Always convert to string for consistent comparison
            relative_name_str = str(relative_name)
            if relative_name_str not in existing_filenames:
                files_to_process.append(file_path)
    
    # Only create targets for folders that have files to process
    targets = list(set([image_to_pickle_name(f, crop_size, model_type) for f in files_to_process]))

    return {
            'file_dep' : files_to_process,
            'targets' : targets,
            'actions':[encode_folder_action],
            'clean':True,

        
    }

def task_cluster_images():
    def cluster_images(dependencies, targets):
        from sklearn.cluster import KMeans
        import pandas as pd
        import numpy as np

        if not dependencies:
            print("No .npz files found to cluster.")
            return
        npz_path = Path(dependencies[0])
        try:
            data = np.load(npz_path, allow_pickle=True)
            embeddings = data['embeddings']
            filenames = data['filenames']
            if len(embeddings) > 2:
                num_clusters = min(5, len(embeddings) // 2)
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                labels = kmeans.fit_predict(embeddings)

                df = pd.DataFrame({
                    'filename': filenames,
                    'cluster': labels
                })
                output_csv = targets[0]

                print(f"Clustered {len(embeddings)} embeddings into {num_clusters} clusters. Saved to {output_csv}")
            else:
                df = pd.DataFrame(columns=['filename', 'cluster'])                
            df.to_csv(output_csv, index=False)                

        except Exception as e:
            print(f"Error processing {npz_path}: {e}")


    input_dir = cfg.get_url('encode_input_dir')
    # get all jpgs in subfolders
    files = list(input_dir.rglob(f'**/*.npz')) 
    for file in files:
        targets = file.parent / f"{file.stem}_clustered.csv"
        yield {
                    'name' : targets,
                    'file_dep' : [file],
                    'targets' : [targets],
                    'actions':[cluster_images],
                    'clean':True,
                    'uptodate': [True],

                }
    
# def task_plot_dendrograms():
#     def plot_dendrogram(dependencies, targets):
#         import numpy as np
#         from sklearn.metrics.pairwise import cosine_similarity
#         from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
#         import matplotlib.pyplot as plt
        
#         npz_path = Path(dependencies[0])
        
#         try:
#             data = np.load(npz_path, allow_pickle=True)
#             embeddings = data['embeddings']
#             filenames = data['filenames']
            
#             if len(embeddings) < 2:
#                 print(f"Skipping {npz_path.stem}: too few samples ({len(embeddings)})")
#                 # Create empty marker file
#                 with open(targets[0], 'w') as f:
#                     f.write(f"Skipped {npz_path.stem}: too few samples")
#                 return
            
#             # Step 1: Compute cosine distance (1 - similarity)
#             cos_sim = cosine_similarity(embeddings)
#             cos_dist = 1 - cos_sim
            
#             # Step 2: Hierarchical clustering
#             Z = linkage(cos_dist, method="average")  # "average" works well for cosine
            
#             # Step 3: Plot dendrogram
#             plt.figure(figsize=(12, 6))
#             dendrogram(Z, labels=np.arange(len(embeddings)))
#             plt.title(f"Hierarchical Clustering Dendrogram - {npz_path.stem}")
#             plt.xlabel("Image index")
#             plt.ylabel("Cosine distance")
            
#             # Save plot
#             plot_path = targets[0]
#             plt.savefig(plot_path, dpi=150, bbox_inches='tight')
#             plt.close()  # Close figure to free memory
            
#             # Step 4: Optionally cut into flat clusters and save info
#             labels = fcluster(Z, t=1.5, criterion="distance")
            
#             # Save cluster info to text file
#             info_path = Path(plot_path).with_suffix('.txt')
#             with open(info_path, 'w') as f:
#                 f.write(f"Hierarchical clustering results for {npz_path.stem}\n")
#                 f.write(f"Number of images: {len(embeddings)}\n")
#                 f.write(f"Number of clusters (distance < 0.3): {len(np.unique(labels))}\n")
#                 f.write(f"Cluster labels: {labels.tolist()}\n")
#                 f.write("\nFilename to cluster mapping:\n")
#                 for filename, label in zip(filenames, labels):
#                     f.write(f"{filename}: cluster_{label}\n")
            
#             print(f"Created dendrogram for {npz_path.stem}: {len(embeddings)} images, {len(np.unique(labels))} clusters")
            
#         except Exception as e:
#             print(f"Error creating dendrogram for {npz_path}: {e}")
#             # Create error marker file
#             with open(targets[0], 'w') as f:
#                 f.write(f"Error processing {npz_path.stem}: {e}")

#     input_dir = cfg.get_url('encode_input_dir')
#     output_dir = cfg.get_url('encode_output_dir')
    
#     # Find all .npz files
#     npz_files = list(input_dir.rglob('**/*.npz'))
    
#     for npz_file in npz_files:
#         # Create dendrogram task for each npz file
#         survey_name = npz_file.stem
#         plot_name = f"{survey_name}_dendrogram.png"
#         plot_path = Path(output_dir) / plot_name
#         plot_path.parent.mkdir(exist_ok=True, parents=True)
        
#         yield {
#             'name': f"dendrogram_{survey_name}",
#             'file_dep': [npz_file],
#             'targets': [plot_path],
#             'actions': [plot_dendrogram],
#             'clean': True,
#         }

# def task_organize_clusters():
#     def organize_cluster_images(dependencies, targets):
#         import pandas as pd
#         import shutil
        
#         csv_path = Path(dependencies[0])
        
#         try:
#             # Load cluster CSV
#             df = pd.read_csv(csv_path)
            
#             # Get the base directory (where the images are located)
#             base_dir = csv_path.parent
            
#             # Get output directory from config
#             output_dir = Path(cfg.get('encode_output_dir'))
            
#             # Create cluster directories for this survey/folder
#             survey_name = csv_path.stem.replace('_clustered', '')
#             cluster_base_dir = output_dir / f"{survey_name}_clusters"
#             cluster_base_dir.mkdir(exist_ok=True)
            
#             # Group by cluster
#             clusters = df.groupby('cluster')
            
#             for cluster_id, cluster_df in clusters:
#                 # Create cluster directory
#                 cluster_dir = cluster_base_dir / f"cluster_{cluster_id}"
#                 cluster_dir.mkdir(exist_ok=True)
                
#                 # Copy images to cluster directory
#                 copied_count = 0
#                 for filename in cluster_df['filename']:
#                     # Construct source path
#                     src_path = base_dir / filename
                    
#                     if src_path.exists():
#                         # Create destination path (keep original filename)
#                         dst_path = cluster_dir / src_path.name
                        
#                         # Copy file if it doesn't exist or is newer
#                         if not dst_path.exists() or src_path.stat().st_mtime > dst_path.stat().st_mtime:
#                             shutil.copy2(src_path, dst_path)
#                             copied_count += 1
                
#                 print(f"Cluster {cluster_id}: copied {copied_count} images to {cluster_dir}")
            
#             # Create completion marker
#             marker_file = targets[0]
#             with open(marker_file, 'w') as f:
#                 f.write(f"Organized clusters for {survey_name} at {cluster_base_dir}")
                
#         except Exception as e:
#             print(f"Error organizing clusters from {csv_path}: {e}")

#     input_dir = cfg.get_url('encode_input_dir')
#     output_dir = cfg.get_url('encode_output_dir')
    
#     # Find all cluster CSV files
#     cluster_files = list(input_dir.rglob('**/*_clustered.csv'))
    
#     for csv_file in cluster_files:
#         # Create organize task for each cluster CSV
#         survey_name = csv_file.stem.replace('_clustered', '')
#         marker_name = f"{survey_name}_clusters_organized.txt"
#         marker_path = Path(output_dir) / marker_name
#         marker_path.parent.mkdir(exist_ok=True, parents=True)
        
#         yield {
#             'name': f"organize_clusters_{survey_name}",
#             'file_dep': [csv_file],
#             'targets': [marker_path],
#             'actions': [organize_cluster_images],
#             'clean': True,
#         }




@app.command('encode')
def doit_encode(
    ctx: typer.Context,
    input_dir: Path = typer.Argument(..., help="Directory containing image folders to process"),
    pattern: str = typer.Option("*.jpg", help="Image pattern to match"),
    doit_db: str = typer.Option(".doit-db", help="Path to doit database file"),
    batch_size: int = typer.Option(128, help="Batch size for processing images"),
    crop_size: int = typer.Option(0, help="Crop size for images to encode"),
    model_type: str = typer.Option('clip', help="Model to use: 'clip', 'dinov3', or 'dinov3_timm'")
):
    """
    Use doit to encode images in folders (for dependency tracking and incremental builds).
    """
    input_dir = Path(input_dir)
    doit_db = input_dir / doit_db 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Print device information
    print(f"🚀 Using device: {device.upper()}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name()}")
    else:
        print("   Note: CUDA not available, using CPU (encoding will be slower)")
    # Validate model type
    valid_models = ['clip', 'dinov3', 'dinov3_timm']
    if model_type not in valid_models:
        typer.echo(f"Error: model_type must be one of {valid_models}")
        raise typer.Exit(1)
    
    # Run doit with custom database location
    cfg.set('encode_input_dir', str(input_dir))
    cfg.set('encode_pattern', pattern)
    cfg.set('encode_batch_size', batch_size)
    cfg.set('encode_crop_size', crop_size)
    cfg.set('encode_model_type', model_type)
  
    DoitMain(ModuleTaskLoader(globals())).run(ctx.args + ['--db-file', str(doit_db), 'encode_folders'])



@app.command()
def inspect(
    npz_file: Path = typer.Option(..., help="Path to .npz file to inspect")
):
    """
    Inspect the contents of an embeddings .npz file.
    """
    if not npz_file.exists():
        typer.echo(f"Error: File {npz_file} does not exist")
        raise typer.Exit(1)
    
    try:
        data = np.load(npz_file, allow_pickle=True)
        
        typer.echo(f"Contents of {npz_file}:")
        typer.echo(f"Keys: {list(data.keys())}")
        
        if 'embeddings' in data:
            embeddings = data['embeddings']
            typer.echo(f"Embeddings shape: {embeddings.shape}")
            typer.echo(f"Embeddings dtype: {embeddings.dtype}")
        
        if 'filenames' in data:
            filenames = data['filenames']
            typer.echo(f"Number of files: {len(filenames)}")
            typer.echo(f"First 5 filenames: {filenames[:5]}")
        
        if 'dimensions' in data:
            dimensions = data['dimensions']
            typer.echo(f"Dimensions shape: {dimensions.shape if dimensions is not None else 'None'}")
            if dimensions is not None and len(dimensions) > 0:
                typer.echo(f"First 5 dimensions: {dimensions[:5]}")
        
        if 'crop_size' in data:
            crop_size_data = data['crop_size']
            typer.echo(f"Crop size: {crop_size_data[0]} (array of {len(crop_size_data)} values)")
        
        if 'model_type' in data:
            model_type_data = data['model_type']
            typer.echo(f"Model type: {model_type_data[0]} (array of {len(model_type_data)} values)")
        
        if 'folder_name' in data:
            typer.echo(f"Folder name: {data['folder_name']}")
        
        if 'folder_path' in data:
            typer.echo(f"Folder path: {data['folder_path']}")
            
    except Exception as e:
        typer.echo(f"Error reading {npz_file}: {e}")
        raise typer.Exit(1)

def load_image_dimensions(npz_path):
    """Load image dimensions from NPZ file"""
    if not npz_path.exists():
        return {}
    
    try:
        data = np.load(npz_path, allow_pickle=True)
        if 'dimensions' in data and 'filenames' in data:
            # Ensure all filenames are strings to avoid Path object issues
            filenames = [str(f) for f in data['filenames']]
            dimensions_dict = dict(zip(filenames, data['dimensions']))
            return dimensions_dict
        else:
            # For backward compatibility with old NPZ files without dimensions
            return {}
    except Exception as e:
        print(f"Error loading dimensions from {npz_path}: {e}")
        return {}

def get_image_dimensions_cached(image_path, crop_size=0, model_type='clip'):
    """Get image dimensions from cached NPZ file if available, otherwise open image"""
    from pathlib import Path
    
    def image_to_pickle_name(image_path, crop_size, model_type='clip'):
        return image_path.parent / f'encoding_{model_type}_{crop_size:03d}.npz'
    
    image_path = Path(image_path)
    npz_path = image_to_pickle_name(image_path, crop_size, model_type)
    relative_name = image_path.relative_to(npz_path.parent)
    
    # Try to load from cache first
    dimensions_dict = load_image_dimensions(npz_path)
    if str(relative_name) in dimensions_dict:
        return tuple(dimensions_dict[str(relative_name)])
    
    # Fallback to opening the image (backward compatibility)
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            return img.size
    except Exception as e:
        print(f"Error getting dimensions for {image_path}: {e}")
        return (0, 0)

def concatenate_npz_files(npz_paths, output_path):
    """Concatenate multiple compatible NPZ files into one"""
    if not npz_paths:
        print("No NPZ files provided for concatenation")
        return False
    
    all_embeddings = {}
    all_dimensions = {}
    reference_metadata = None
    
    for npz_path in npz_paths:
        embeddings, dimensions, metadata = load_existing_embeddings(npz_path)
        
        if not embeddings:
            print(f"Skipping empty file: {npz_path}")
            continue
        
        # Check compatibility
        if reference_metadata is None:
            reference_metadata = metadata
        else:
            if metadata.get('crop_size') != reference_metadata.get('crop_size'):
                print(f"Error: crop_size mismatch in {npz_path}")
                return False
            if metadata.get('model_type') != reference_metadata.get('model_type'):
                print(f"Error: model_type mismatch in {npz_path}")
                return False
        
        # Add embeddings and dimensions (check for duplicates)
        for filename, embedding in embeddings.items():
            if filename in all_embeddings:
                print(f"Warning: duplicate filename '{filename}' in {npz_path}, skipping")
                continue
            all_embeddings[filename] = embedding
            if dimensions and filename in dimensions:
                all_dimensions[filename] = dimensions[filename]
        
        print(f"Added {len(embeddings)} embeddings from {npz_path}")
    
    if not all_embeddings:
        print("No embeddings to save after concatenation")
        return False
    
    # Save concatenated file
    save_folder_embeddings(
        output_path, 
        all_embeddings, 
        all_dimensions if all_dimensions else None,
        reference_metadata.get('crop_size', 0),
        reference_metadata.get('model_type', 'clip')
    )
    
    print(f"Successfully concatenated {len(npz_paths)} files into {output_path}")
    print(f"Total embeddings: {len(all_embeddings)}")
    return True

@app.command()
def concatenate(
    output_file: Path = typer.Argument(..., help="Output NPZ file path"),
    input_files: list[Path] = typer.Argument(..., help="Input NPZ files to concatenate")
):
    """
    Concatenate multiple compatible NPZ embedding files into one.
    All files must have the same crop_size and model_type.
    """
    # Validate input files
    missing_files = [f for f in input_files if not f.exists()]
    if missing_files:
        typer.echo(f"Error: Missing files: {missing_files}")
        raise typer.Exit(1)
    
    if len(input_files) < 2:
        typer.echo("Error: Need at least 2 files to concatenate")
        raise typer.Exit(1)
    
    # Perform concatenation
    success = concatenate_npz_files(input_files, output_file)
    
    if success:
        typer.echo(f"✅ Successfully concatenated {len(input_files)} files into {output_file}")
    else:
        typer.echo("❌ Concatenation failed")
        raise typer.Exit(1)

if __name__ == "__main__":
    app()
