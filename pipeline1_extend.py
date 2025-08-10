import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import json
from datetime import datetime
import time
import sys
import tempfile
import traceback
import importlib.util

# Import UNet and Voronoi modules
from Unet_extend import UNetProcessor
from Voronoi_extend import VoronoiPlanner
from Boundry_extend import BoundaryProcessor
from urban_3d_generator import Urban3DGenerator, run_urban_generator  # Import for 2D visualization

# Import 3D_gen.py as a module
def import_3d_gen_module(file_path="3D_gen.py"):
    """Import the 3D_gen.py file as a module"""
    try:
        module_name = "city_3d_generator"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"Error importing 3D_gen module: {e}")
        traceback.print_exc()
        return None

# ==========================================
# HELPER FUNCTIONS FOR DATA MANAGEMENT
# ==========================================
def create_structured_output_directory(base_dir):
    """Create a structured output directory with subdirectories for different data types"""
    # Create main subdirectories
    user_inputs_dir = os.path.join(base_dir, "1_user_inputs")
    unet_features_dir = os.path.join(base_dir, "2_unet_features")
    voronoi_outputs_dir = os.path.join(base_dir, "3_voronoi_outputs")
    overall_outputs_dir = os.path.join(base_dir, "4_overall_outputs")
    urban_2d_dir = os.path.join(base_dir, "5_urban_2d")  # Directory for 2D building visualizations
    urban_3d_dir = os.path.join(base_dir, "6_3d_city")  # New directory for 3D visualization
    
    # Create directories
    for directory in [user_inputs_dir, unet_features_dir, voronoi_outputs_dir, 
                     overall_outputs_dir, urban_2d_dir, urban_3d_dir]:
        os.makedirs(directory, exist_ok=True)
    
    return {
        "base_dir": base_dir,
        "user_inputs": user_inputs_dir,
        "unet_features": unet_features_dir,
        "voronoi_outputs": voronoi_outputs_dir,
        "overall_outputs": overall_outputs_dir,
        "urban_2d": urban_2d_dir,
        "urban_3d": urban_3d_dir
    }

def enhanced_convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    Enhanced to handle more data types and special cases.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        # Special handling for ndarray - check if it contains complex objects
        if obj.dtype.kind in ['O']:
            # Object array - convert each element recursively
            return [enhanced_convert_numpy_types(item) for item in obj]
        else:
            # Numeric array - convert to list
            return obj.tolist()
    elif isinstance(obj, dict):
        return {key: enhanced_convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [enhanced_convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    else:
        # Try to convert to string for unknown types
        try:
            return str(obj)
        except:
            return "Unserializable object"

def save_json_data(data, directory, filename, prefix=""):
    """Save data as JSON to the specified directory"""
    try:
        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)
        
        # Add prefix to filename if provided
        if prefix and not filename.startswith(prefix):
            filename = f"{prefix}_{filename}"
        
        # Ensure filename has .json extension
        if not filename.endswith(".json"):
            filename += ".json"
            
        # Complete file path
        filepath = os.path.join(directory, filename)
        
        # Convert data to JSON-serializable format
        converted_data = enhanced_convert_numpy_types(data)
        
        # Write to file
        with open(filepath, 'w') as f:
            json.dump(converted_data, f, indent=2)
            
        return filepath
    except Exception as e:
        print(f"Error saving JSON data: {e}")
        traceback.print_exc()
        return None

# Function to generate 2D building visualization from Voronoi output
def generate_2d_urban_visualization(voronoi_image_path, output_dir):
    """
    Generate 2D urban visualization with buildings from a Voronoi diagram image
    
    Args:
        voronoi_image_path: Path to the voronoi diagram image
        output_dir: Directory to save 2D outputs
    
    Returns:
        Dictionary with paths to generated 2D output files
    """
    try:
        # Set up generator for 2D visualization
        st.info("Generating 2D urban visualization from Voronoi plan...")
        
        # Run the generator with the Voronoi image
        # Note: Removed the 'only_2d' parameter which was causing the error
        output_files = run_urban_generator(
            image_path=voronoi_image_path,
            output_dir=output_dir,
            export_obj=False,  # Don't export 3D model
            show_vis=False     # Don't show visualizations in Streamlit environment
        )
        
        if not output_files or 'vis_2d' not in output_files:
            st.error("Failed to generate 2D urban visualization.")
            return None
        
        # Create metadata for JSON
        visualization_metadata = {
            "timestamp": datetime.now().isoformat(),
            "source_image": voronoi_image_path,
            "vis_2d": output_files.get('vis_2d')
        }
        
        # Save metadata
        metadata_path = save_json_data(
            visualization_metadata,
            output_dir,
            f"urban_2d_metadata_{datetime.now().strftime('%H%M%S')}.json"
        )
        
        return {
            "metadata": metadata_path,
            "files": output_files
        }
        
    except Exception as e:
        st.error(f"Error generating 2D visualization: {e}")
        traceback.print_exc()
        return None

def generate_3d_city_visualization(voronoi_image_path, output_dir):
    """
    Generate 3D city visualization using 3D_gen.py
    
    Args:
        voronoi_image_path: Path to the Voronoi diagram image from urban_3d_generator
        output_dir: Directory to save 3D outputs
    
    Returns:
        Dictionary with paths to generated 3D output files
    """
    try:
        # Import the 3D_gen module
        city_3d_gen = import_3d_gen_module()
        if city_3d_gen is None:
            st.error("Failed to import 3D_gen.py module")
            return None
        
        # Set output HTML file path
        html_output_path = os.path.join(output_dir, f"3d_city_{datetime.now().strftime('%H%M%S')}.html")
        
        # Set parameters for 3D generation
        max_buildings = 300
        max_trees = 300
        show_viz = False  # Don't show intermediate visualizations in Streamlit environment
        
        st.info(f"Generating 3D city visualization. This may take a few minutes...")
        
        # Call the 3D generation function from 3D_gen.py
        building_models, trees, fig = city_3d_gen.generate_city_from_voronoi_with_genai(
            voronoi_image_path,
            output_path=html_output_path,
            show_viz=show_viz,
            max_buildings=max_buildings,
            max_trees=max_trees
        )
        
        # Create metadata for JSON
        visualization_metadata = {
            "timestamp": datetime.now().isoformat(),
            "source_image": voronoi_image_path,
            "html_output": html_output_path,
            "building_count": len(building_models),
            "tree_count": len(trees)
        }
        
        # Save metadata
        metadata_path = save_json_data(
            visualization_metadata,
            output_dir,
            f"3d_city_metadata_{datetime.now().strftime('%H%M%S')}.json"
        )
        
        return {
            "metadata": metadata_path,
            "html_output": html_output_path,
            "building_count": len(building_models),
            "tree_count": len(trees),
            "figure": fig  # Plotly figure object
        }
        
    except Exception as e:
        st.error(f"Error generating 3D visualization: {e}")
        traceback.print_exc()
        return None

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temp directory and return path"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving uploaded file: {e}")
        return None

def create_timestamp_folder():
    """Create a timestamped folder for outputs with structured subdirectories"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_path = os.path.join(OUTPUT_DIR, timestamp)
    
    # Create structured directories using the helper function
    return create_structured_output_directory(folder_path)

# Set page configuration
st.set_page_config(
    page_title="UDGAN - Urban Design Generator",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths
OUTPUT_DIR = "udgan_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize session state variables
if 'output_dirs' not in st.session_state:
    st.session_state.output_dirs = create_timestamp_folder()
if 'image_path' not in st.session_state:
    st.session_state.image_path = None
if 'segmentation_results' not in st.session_state:
    st.session_state.segmentation_results = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'voronoi_data' not in st.session_state:
    st.session_state.voronoi_data = None
if 'urban_plan' not in st.session_state:
    st.session_state.urban_plan = None
if 'is_sketch_input' not in st.session_state:
    st.session_state.is_sketch_input = False
if 'urban_2d_outputs' not in st.session_state:
    st.session_state.urban_2d_outputs = None
if 'city_3d_outputs' not in st.session_state:  # New state variable for 3D outputs
    st.session_state.city_3d_outputs = None

# Initialize the UNet, Voronoi, and Boundary processors
@st.cache_resource
def load_processors():
    unet = UNetProcessor()
    voronoi = VoronoiPlanner()
    boundary = BoundaryProcessor()
    return unet, voronoi, boundary

unet, voronoi, boundary_processor = load_processors()

# App title and description
st.title("🏙️ UDGAN - Urban Design Generative Adversarial Network")
st.markdown("""
This application implements the UDGAN framework for urban design generation:
1. **UNet Model** processes urban imagery to extract features
2. **Voronoi Planner** creates space partitioning for urban zoning
3. **UDGAN Core** (preview) generates optimized urban layouts
4. **3D Visualization** creates detailed 3D city models
""")

# Create sidebar for settings and controls
st.sidebar.title("Settings & Processing")

# Add information about output directory
st.sidebar.subheader("Output Directory")
st.sidebar.info(f"Results will be saved to: {st.session_state.output_dirs['base_dir']}")
if st.sidebar.button("Create New Output Directory"):
    st.session_state.output_dirs = create_timestamp_folder()
    st.sidebar.success(f"New directory created: {st.session_state.output_dirs['base_dir']}")

# Area scale setting
st.sidebar.subheader("Area Scale Settings")
area_scale = st.sidebar.slider(
    "Scale Factor (sq ft per pixel)",
    min_value=0.1,
    max_value=10.0,
    value=1.0,
    step=0.1,
    help="Set the scale factor for area calculations"
)

# Initialize UNet with scale factor
unet.set_scale_factor(area_scale)

# Add image upload
st.sidebar.subheader("Input Image")
uploaded_file = st.sidebar.file_uploader(
    "Upload urban area image or boundary sketch",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
    help="Upload an image of an urban area or a hand-drawn boundary sketch for processing"
)

# Add sample image generation option
generate_sample = st.sidebar.button("Generate Sample Image")

# Add new section for boundary sketches
st.sidebar.subheader("Sketch Processing")
direct_to_voronoi = st.sidebar.checkbox(
    "Direct Sketch to Voronoi",
    value=False,
    help="When enabled, hand-drawn boundary sketches will bypass UNet and go directly to the Voronoi planner"
)

# Boundary processing options
if st.sidebar.checkbox("Advanced Boundary Options", value=False):
    st.sidebar.subheader("Boundary Processing Settings")
    
    text_removal = st.sidebar.checkbox("Remove Text from Sketch", value=True,
              help="Remove text measurements and labels from sketch")
    boundary_processor.text_detection_enabled = text_removal
    
    if text_removal:
        text_threshold = st.sidebar.slider("Text Detection Sensitivity", 50, 200, 100, 
                        help="Higher values detect more text regions")
        boundary_processor.text_area_threshold = text_threshold
    
    simplify_boundary = st.sidebar.checkbox("Simplify Boundary", value=True,
                  help="Simplify boundary by removing unnecessary points")
    boundary_processor.simplification_enabled = simplify_boundary
    
    if simplify_boundary:
        simplification_level = st.sidebar.slider("Simplification Level", 1, 20, 5,
                          help="Higher values produce simpler boundaries")
        boundary_processor.simplification_tolerance = simplification_level
        
        angle_threshold = st.sidebar.slider("Corner Angle Threshold", 120, 175, 160,
                     help="Angles below this threshold are considered corners")
        boundary_processor.angle_threshold = angle_threshold

# Main content area with tabs - add 3D Visualization tab
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 UNet Segmentation",
    "🗺️ Voronoi Planning",
    "📈 Urban Metrics",
    "🏙️ 3D Visualization"  # New tab
])

# Process image when uploaded or sample generated
if uploaded_file is not None:
    with st.spinner("Processing uploaded image..."):
        # Save uploaded file
        file_path = save_uploaded_file(uploaded_file)
        st.session_state.image_path = file_path
        
        # Save initial user inputs (image metadata)
        image_metadata = {
            "filename": uploaded_file.name,
            "size_bytes": uploaded_file.size,
            "timestamp": datetime.now().isoformat(),
            "uploaded_image_path": file_path,
            "area_scale": area_scale
        }
        
        metadata_path = save_json_data(
            image_metadata,
            st.session_state.output_dirs["user_inputs"],
            f"image_metadata_{datetime.now().strftime('%H%M%S')}.json"
        )
        
        # Check if it's a sketch input
        img = cv2.imread(file_path)
        if img is not None:
            st.session_state.is_sketch_input = boundary_processor.is_sketch_input(img)
            if st.session_state.is_sketch_input:
                st.sidebar.success("✅ Detected hand-drawn sketch input")
            
        # Display image in UNet tab
        with tab1:
            st.subheader("Input Image")
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            if st.session_state.is_sketch_input:
                st.info("📝 This appears to be a hand-drawn sketch. Specialized boundary extraction will be used.")

elif generate_sample:
    with st.spinner("Generating sample image..."):
        # Generate and save sample
        sample_data = unet.generate_sample_data(
            num_samples=1,
            output_dir=st.session_state.output_dirs["base_dir"]
        )
        if sample_data:
            sample_path = list(sample_data.values())[0]
            st.session_state.image_path = sample_path
            st.session_state.is_sketch_input = False
            
            # Save sample metadata
            sample_metadata = {
                "filename": os.path.basename(sample_path),
                "timestamp": datetime.now().isoformat(),
                "generated_image_path": sample_path,
                "is_sample": True,
                "area_scale": area_scale
            }
            
            metadata_path = save_json_data(
                sample_metadata,
                st.session_state.output_dirs["user_inputs"],
                f"sample_metadata_{datetime.now().strftime('%H%M%S')}.json"
            )
            
            # Display sample image in UNet tab
            with tab1:
                st.subheader("Sample Image")
                st.image(sample_path, caption="Generated Sample", use_column_width=True)
        else:
            st.error("Failed to generate sample image")

# UNet Processing Section
with tab1:
    st.header("UNet Urban Segmentation")
    if st.session_state.image_path:
        # Process image
        unet_col1, unet_col2 = st.columns(2)
        
        with unet_col1:
            if st.button("Process Image with UNet"):
                with st.spinner("Running UNet segmentation..."):
                    try:
                        st.session_state.segmentation_results = unet.segment_image(st.session_state.image_path)
                        
                        # Save segmentation data (non-visual parts only)
                        segmentation_data = {
                            "is_sketch": st.session_state.segmentation_results.get("is_sketch", False),
                            "boundary": enhanced_convert_numpy_types(
                                st.session_state.segmentation_results.get("boundary", [])
                            ),
                            "path": st.session_state.image_path,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        segmentation_path = save_json_data(
                            segmentation_data,
                            st.session_state.output_dirs["unet_features"],
                            f"segmentation_results_{datetime.now().strftime('%H%M%S')}.json"
                        )
                        
                        st.success(f"Segmentation completed and saved!")
                        
                        # If it's a sketch, update the flag to ensure we know it throughout the pipeline
                        if "is_sketch" in st.session_state.segmentation_results:
                            st.session_state.is_sketch_input = st.session_state.segmentation_results["is_sketch"]
                            
                    except Exception as e:
                        st.error(f"Error in segmentation: {e}")
                        traceback.print_exc()
                        
            # Add specific button for sketch boundary extraction
            if st.session_state.is_sketch_input:
                if st.button("Extract Boundary from Sketch"):
                    with st.spinner("Extracting boundary from sketch..."):
                        try:
                            img = cv2.imread(st.session_state.image_path)
                            
                            # Configure boundary processor parameters based on UI settings                            
                            boundary_extraction = boundary_processor.extract_boundary_from_sketch(img)
                            
                            if boundary_extraction and len(boundary_extraction["boundary"]) >= 3:
                                st.success("Successfully extracted boundary from sketch!")
                                
                                # Save boundary extraction data
                                boundary_data = {
                                    "boundary": enhanced_convert_numpy_types(boundary_extraction["boundary"]),
                                    "corners": enhanced_convert_numpy_types(boundary_extraction.get("corners", [])),
                                    "timestamp": datetime.now().isoformat()
                                }
                                
                                boundary_path = save_json_data(
                                    boundary_data,
                                    st.session_state.output_dirs["unet_features"],
                                    f"boundary_extraction_{datetime.now().strftime('%H%M%S')}.json"
                                )
                                
                                # Store the results in the session state
                                if st.session_state.segmentation_results is None:
                                    # Create minimal segmentation results with the boundary
                                    st.session_state.segmentation_results = {
                                        "is_sketch": True,
                                        "boundary": boundary_extraction["boundary"],
                                        "boundary_extraction": boundary_extraction,
                                        "original": img,
                                        "path": st.session_state.image_path
                                    }
                                else:
                                    # Update existing results
                                    st.session_state.segmentation_results["is_sketch"] = True
                                    st.session_state.segmentation_results["boundary"] = boundary_extraction["boundary"]
                                    st.session_state.segmentation_results["boundary_extraction"] = boundary_extraction
                            else:
                                st.error("Failed to extract boundary from sketch. Try adjusting the image or using a clearer boundary.")
                        except Exception as e:
                            st.error(f"Error extracting boundary: {e}")
                            traceback.print_exc()
        
        with unet_col2:
            if st.button("Extract Urban Features"):
                if st.session_state.segmentation_results is None:
                    st.warning("Please process the image with UNet first")
                else:
                    with st.spinner("Extracting features..."):
                        try:
                            st.session_state.features = unet.extract_features(st.session_state.segmentation_results)
                            
                            # Save features to JSON
                            features_path = save_json_data(
                                st.session_state.features,
                                st.session_state.output_dirs["unet_features"],
                                f"features_{datetime.now().strftime('%H%M%S')}.json"
                            )
                            
                            st.success(f"Features extracted and saved!")
                        except Exception as e:
                            st.error(f"Error extracting features: {e}")
                            traceback.print_exc()
            
            # Add option to skip to Voronoi for sketch inputs
            if st.session_state.is_sketch_input and direct_to_voronoi:
                if st.button("Skip to Voronoi Planning"):
                    if st.session_state.segmentation_results is None or "boundary" not in st.session_state.segmentation_results:
                        st.warning("Please extract the boundary from the sketch first")
                    else:
                        with st.spinner("Preparing data for Voronoi..."):
                            try:
                                # Create minimal voronoi data with just the boundary
                                boundary = st.session_state.segmentation_results["boundary"]
                                st.session_state.voronoi_data = {
                                    "boundary": boundary,
                                    "building_centroids": [],
                                    "road_intersections": [],
                                    "features": {"scale_factor": area_scale}
                                }
                                
                                # Save voronoi prep data
                                voronoi_prep_path = save_json_data(
                                    st.session_state.voronoi_data,
                                    st.session_state.output_dirs["voronoi_outputs"],
                                    f"voronoi_prep_sketch_{datetime.now().strftime('%H%M%S')}.json"
                                )
                                
                                st.success(f"Boundary prepared for Voronoi planning! Please go to the Voronoi Planning tab.")
                            except Exception as e:
                                st.error(f"Error preparing boundary for Voronoi: {e}")
                                traceback.print_exc()
        
        # Show segmentation results if they exist
        if st.session_state.segmentation_results is not None:
            st.subheader("Segmentation Results")
            
            # Instead of visualizing boundary extraction, just show a simple message
            if st.session_state.is_sketch_input:
                st.info("Boundary extracted and saved in the backend.")
                
                # For sketch inputs, show different options
                if st.button("Process Boundary for Voronoi"):
                    with st.spinner("Preparing boundary for Voronoi planning..."):
                        try:
                            boundary = st.session_state.segmentation_results.get("boundary", None)
                            if boundary is None or len(boundary) < 3:
                                st.error("Invalid boundary detected. Please try again.")
                            else:
                                # Create basic voronoi data with just the boundary
                                st.session_state.voronoi_data = {
                                    "boundary": boundary,
                                    "building_centroids": [],
                                    "road_intersections": [],
                                    "features": {"scale_factor": area_scale}
                                }
                                
                                # Save voronoi prep data
                                voronoi_prep_path = save_json_data(
                                    st.session_state.voronoi_data,
                                    st.session_state.output_dirs["voronoi_outputs"],
                                    f"voronoi_prep_boundary_{datetime.now().strftime('%H%M%S')}.json"
                                )
                                
                                st.success(f"Boundary prepared for Voronoi planning!")
                        except Exception as e:
                            st.error(f"Error preparing boundary: {e}")
                            traceback.print_exc()
            else:
                # For regular inputs, show normal options
                # Create and display binary masks
                if st.button("Create Binary Masks"):
                    with st.spinner("Creating binary masks..."):
                        binary_masks = unet.create_binary_masks(st.session_state.segmentation_results)
                        # Save and display first mask as example
                        if binary_masks:
                            st.success(f"Created {len(binary_masks)} binary masks")
                            mask_name, mask = list(binary_masks.items())[0]
                            mask_path = os.path.join(
                                st.session_state.output_dirs["base_dir"],
                                f"{mask_name}_mask.png"
                            )
                            cv2.imwrite(mask_path, mask)
                            st.image(mask_path, caption=f"{mask_name.replace('_', ' ').title()} Mask (Example)", 
                                    use_column_width=True)
                            
                            # Save mask data to JSON
                            mask_metadata = {
                                "mask_types": list(binary_masks.keys()),
                                "timestamp": datetime.now().isoformat(),
                                "base_path": st.session_state.output_dirs["base_dir"]
                            }
                            
                            mask_meta_path = save_json_data(
                                mask_metadata,
                                st.session_state.output_dirs["unet_features"],
                                f"binary_masks_metadata_{datetime.now().strftime('%H%M%S')}.json"
                            )
                    
                # Create tensor representation
                if st.button("Create UDGAN Tensor"):
                    with st.spinner("Creating multi-channel tensor..."):
                        tensor = unet.create_tensor_representation(st.session_state.segmentation_results)
                        if tensor is not None:
                            tensor_path = os.path.join(
                                st.session_state.output_dirs["overall_outputs"],
                                f"udgan_tensor_{datetime.now().strftime('%H%M%S')}.npy"
                            )
                            np.save(tensor_path, tensor)
                            
                            # Save tensor metadata
                            tensor_metadata = {
                                "tensor_path": tensor_path,
                                "shape": tensor.shape,
                                "timestamp": datetime.now().isoformat(),
                                "channels": {
                                    "0": "Buildings",
                                    "1": "Roads",
                                    "2": "Green Spaces",
                                    "3": "Water Bodies"
                                }
                            }
                            
                            tensor_meta_path = save_json_data(
                                tensor_metadata,
                                st.session_state.output_dirs["overall_outputs"],
                                f"tensor_metadata_{datetime.now().strftime('%H%M%S')}.json"
                            )
                            
                            st.success(f"UDGAN tensor created and saved!")
                            
                            # Visualize tensor channels
                            fig, axs = plt.subplots(2, 2, figsize=(10, 8))
                            axs = axs.flatten()
                            channel_names = ["Buildings", "Roads", "Green Spaces", "Water Bodies"]
                            for i, (name, ax) in enumerate(zip(channel_names, axs)):
                                ax.imshow(tensor[:, :, i], cmap='viridis')
                                ax.set_title(f"Channel {i}: {name}")
                                ax.axis('off')
                            plt.tight_layout()
                            tensor_viz_path = os.path.join(
                                st.session_state.output_dirs["base_dir"],
                                f"tensor_viz_{datetime.now().strftime('%H%M%S')}.png"
                            )
                            plt.savefig(tensor_viz_path, dpi=300, bbox_inches='tight')
                            plt.close()
                            st.image(tensor_viz_path, caption="Tensor Visualization", use_column_width=True)
        
        # Prepare data for Voronoi if needed
        if st.button("Prepare Voronoi Data"):
            if st.session_state.segmentation_results is None:
                st.warning("Please process the image with UNet first")
            else:
                with st.spinner("Preparing data for Voronoi diagram..."):
                    try:
                        st.session_state.voronoi_data = unet.prepare_for_voronoi(st.session_state.segmentation_results)
                        
                        # Save voronoi preparation data
                        voronoi_prep_path = save_json_data(
                            st.session_state.voronoi_data,
                            st.session_state.output_dirs["voronoi_outputs"],
                            f"voronoi_prep_{datetime.now().strftime('%H%M%S')}.json"
                        )
                        
                        st.success(f"Voronoi preparation data created and saved!")
                    except Exception as e:
                        st.error(f"Error preparing Voronoi data: {e}")
                        traceback.print_exc()
    else:
        st.info("Please upload an image or generate a sample to begin")

# Voronoi Planning Section
with tab2:
    st.header("Voronoi Urban Planning")
    
    # Add boundary verification before Voronoi processing
    if st.session_state.voronoi_data is not None and "boundary" in st.session_state.voronoi_data:
        # Display basic boundary information but not visualization
        st.subheader("Boundary Information")
        boundary = st.session_state.voronoi_data["boundary"]
        if boundary is not None and len(boundary) > 2:
            st.info(f"Boundary with {len(boundary)} points has been extracted and is ready for Voronoi planning.")
            
            # Add manual boundary adjustment options
            st.subheader("Boundary Adjustments")
            
            if st.checkbox("Enable Manual Boundary Adjustments"):
                st.info("You can manually adjust the boundary before proceeding with Voronoi planning.")
                
                # Option to remove vertices
                vertices_to_remove = st.multiselect(
                    "Select vertices to remove (if needed)",
                    options=list(range(len(boundary))),
                    format_func=lambda x: f"Vertex {x}"
                )
                
                # Option to simplify boundary further
                simplify_more = st.slider(
                    "Additional Boundary Simplification",
                    min_value=0.0,
                    max_value=10.0,
                    value=0.0,
                    step=0.5,
                    help="Higher values create a simpler boundary with fewer points"
                )
                
                if st.button("Apply Boundary Adjustments"):
                    with st.spinner("Adjusting boundary..."):
                        try:
                            # Copy the original boundary
                            adjusted_boundary = boundary.copy()
                            
                            # Remove selected vertices
                            if vertices_to_remove:
                                keep_indices = [i for i in range(len(boundary)) if i not in vertices_to_remove]
                                adjusted_boundary = adjusted_boundary[keep_indices]
                            
                            # Apply additional simplification if requested
                            if simplify_more > 0 and len(adjusted_boundary) > 3:
                                from shapely.geometry import Polygon
                                poly = Polygon(adjusted_boundary)
                                if poly.is_valid:
                                    simplified = poly.simplify(simplify_more, preserve_topology=True)
                                    adjusted_boundary = np.array(simplified.exterior.coords)[:-1]
                            
                            # Ensure we still have a valid boundary
                            if len(adjusted_boundary) >= 3:
                                # Update the boundary in the session state
                                st.session_state.voronoi_data["boundary"] = adjusted_boundary
                                
                                if st.session_state.segmentation_results:
                                    st.session_state.segmentation_results["boundary"] = adjusted_boundary
                                
                                # Save adjusted boundary
                                adjusted_boundary_data = {
                                    "boundary": enhanced_convert_numpy_types(adjusted_boundary),
                                    "adjustment_type": "manual",
                                    "vertices_removed": vertices_to_remove,
                                    "additional_simplification": simplify_more,
                                    "timestamp": datetime.now().isoformat()
                                }
                                
                                adj_boundary_path = save_json_data(
                                    adjusted_boundary_data,
                                    st.session_state.output_dirs["voronoi_outputs"],
                                    f"adjusted_boundary_{datetime.now().strftime('%H%M%S')}.json"
                                )
                                
                                st.success(f"Boundary adjusted successfully!")
                                st.experimental_rerun()
                            else:
                                st.error("Resulting boundary would have too few points. Please select fewer vertices to remove.")
                        except Exception as e:
                            st.error(f"Error adjusting boundary: {e}")
                            traceback.print_exc()
            
            # Add option to clear the boundary and start over
            if st.button("Reset Boundary Extraction"):
                st.session_state.voronoi_data = None
                st.session_state.segmentation_results = None
                st.info("Boundary reset. Please return to the UNet tab to extract a new boundary.")
                st.experimental_rerun()
    
    if st.session_state.voronoi_data is None and st.session_state.segmentation_results is not None:
        st.warning("Please prepare Voronoi data in the UNet tab first")
        if st.button("Quick Prepare Voronoi Data"):
            with st.spinner("Preparing data for Voronoi diagram..."):
                try:
                    st.session_state.voronoi_data = unet.prepare_for_voronoi(st.session_state.segmentation_results)
                    
                    # Save voronoi preparation data
                    voronoi_prep_path = save_json_data(
                        st.session_state.voronoi_data,
                        st.session_state.output_dirs["voronoi_outputs"],
                        f"voronoi_prep_quick_{datetime.now().strftime('%H%M%S')}.json"
                    )
                    
                    st.success(f"Voronoi preparation data created and saved!")
                except Exception as e:
                    st.error(f"Error preparing Voronoi data: {e}")
                    traceback.print_exc()
    
    if st.session_state.voronoi_data is not None:
        # Voronoi parameters
        voronoi_col1, voronoi_col2 = st.columns(2)
        
        with voronoi_col1:
            st.subheader("Urban Layout Constraints")
            
            num_zones = st.slider(
                "Number of Urban Zones",
                min_value=5,
                max_value=50,
                value=10,
                step=1,
                help="Number of zones to create in the urban plan"
            )
            
            layout_shape = st.radio(
                "Layout Shape",
                options=["Regular", "Irregular"],
                help="Regular creates a more grid-like urban pattern, while Irregular creates more organic forms"
            )
            
            road_network_type = st.selectbox(
                "Road Network Type",
                options=["Grid", "Organic", "Radial"],
                help="Grid: traditional city grid, Organic: natural growth pattern, Radial: circular pattern emanating from center"
            )
            
            building_density = st.slider(
                "Building Density",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Overall building density from 0 (minimal) to 1 (maximum)"
            )
            
            sustainability_score = st.slider(
                "Sustainability Score",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Influences green space and sustainable features"
            )
        
        with voronoi_col2:
            st.subheader("Zoning Distribution (%)")
            st.info("Set percentages for different zone types (should sum to ~100%)")
            
            # Zone type sliders
            residential = st.slider("Residential", 0, 100, 50, 5)
            commercial = st.slider("Commercial", 0, 100, 20, 5)
            green = st.slider("Green Space", 0, 100, 15, 5)
            mixed = st.slider("Mixed Use", 0, 100, 10, 5)
            civic = st.slider("Civic", 0, 100, 5, 5)
            
            # Calculate the sum and normalize if needed
            zone_sum = residential + commercial + green + mixed + civic
            if zone_sum > 0:
                st.metric("Total Percentage", f"{zone_sum}%", 
                         delta="0%" if zone_sum == 100 else f"{zone_sum - 100}%")
                
                # Create zoning distribution
                zoning_distribution = {
                    "Residential": residential,
                    "Commercial": commercial,
                    "Green": green,
                    "Mixed": mixed,
                    "Civic": civic
                }
                
                if zone_sum != 100:
                    st.warning(f"Percentages sum to {zone_sum}%. They will be normalized to 100%.")
            else:
                st.error("At least one zone type must have a non-zero percentage")
        
        # Create urban plan button
        if st.button("Generate Urban Plan"):
            with st.spinner("Generating urban plan..."):
                try:
                    # Create urban layout constraints
                    constraints = {
                        "area_scale": area_scale,
                        "layout_shape": layout_shape,
                        "building_density": building_density,
                        "building_count": int(num_zones * 3),  # Approximate building count
                        "road_network_type": road_network_type,
                        "sustainability_score": sustainability_score,
                        "zoning_distribution": zoning_distribution,
                        "num_zones": num_zones,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Save user constraints
                    constraints_path = save_json_data(
                        constraints,
                        st.session_state.output_dirs["user_inputs"],
                        f"urban_constraints_{datetime.now().strftime('%H%M%S')}.json"
                    )
                    
                    # Set Voronoi planner parameters
                    voronoi.set_scale(area_scale)
                    voronoi.set_constraints(constraints)
                    
                    # Extract data from UNet voronoi preparation
                    boundary = st.session_state.voronoi_data.get("boundary")
                    if boundary is None or len(boundary) < 3:
                        st.error("Invalid boundary data. Please try again.")
                    else:
                        # Generate urban plan
                        urban_plan = voronoi.process_and_plan(
                            st.session_state.image_path,
                            st.session_state.output_dirs["base_dir"],  # Use base_dir for image outputs
                            num_zones=num_zones,
                            pixels_to_meters=area_scale
                        )
                        
                        if urban_plan:
                            # Store in session state
                            st.session_state.urban_plan = urban_plan
                            
                            # Save Voronoi plan data
                            voronoi_plan_path = save_json_data(
                                urban_plan,
                                st.session_state.output_dirs["voronoi_outputs"],
                                f"voronoi_plan_{datetime.now().strftime('%H%M%S')}.json"
                            )
                            
                            st.success(f"Urban plan created and saved!")
                            
                            # Display the urban plan visualization
                            if 'visualization_output' in urban_plan and urban_plan['visualization_output']:
                                st.image(urban_plan['visualization_output'], 
                                        caption="Generated Urban Plan", use_column_width=True)
                                
                                # Generate 2D urban visualization with buildings
                                with st.spinner("Generating 2D visualization with buildings..."):
                                    try:
                                        urban_2d_outputs = generate_2d_urban_visualization(
                                            urban_plan['visualization_output'],
                                            st.session_state.output_dirs["urban_2d"]
                                        )
                                        
                                        if urban_2d_outputs and 'files' in urban_2d_outputs and 'vis_2d' in urban_2d_outputs['files']:
                                            st.session_state.urban_2d_outputs = urban_2d_outputs
                                            st.success("2D urban visualization with buildings generated successfully!")
                                            # Display the 2D visualization
                                            st.image(urban_2d_outputs['files']['vis_2d'], 
                                                    caption="2D Urban Layout with Buildings", use_column_width=True)
                                        else:
                                            st.warning("Could not generate 2D urban visualization. Showing base Voronoi plan only.")
                                    except Exception as e:
                                        st.error(f"Error generating 2D urban visualization: {e}")
                                        traceback.print_exc()
                                
                        else:
                            st.error("Failed to create urban plan")
                except Exception as e:
                    st.error(f"Error generating urban plan: {e}")
                    traceback.print_exc()
    elif st.session_state.image_path is None:
        st.info("Please upload an image or generate a sample in the UNet tab first")
    else:
        st.info("Process the image with UNet and prepare Voronoi data first")

# Urban Metrics Section
with tab3:
    st.header("Urban Metrics & Analysis")
    
    if st.session_state.features is not None:
        st.subheader("Morphological Parameters")
        
        # Create metrics dashboard
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        with metrics_col1:
            st.metric("DN (Density)", f"{st.session_state.features.get('DN', 0):.3f}")
            st.metric("FAR (Floor Area Ratio)", f"{st.session_state.features.get('FAR', 0):.3f}")
            st.metric("AH (Average Height)", f"{st.session_state.features.get('AH', 0):.1f} m")
        
        with metrics_col2:
            st.metric("GP (Green Proportion)", f"{st.session_state.features.get('GP', 0):.3f}")
            st.metric("WP (Water Proportion)", f"{st.session_state.features.get('WP', 0):.3f}")
            st.metric("HP (Hardscape Proportion)", f"{st.session_state.features.get('HP', 0):.3f}")
        
        with metrics_col3:
            st.metric("Building Count", f"{st.session_state.features.get('building_count', 0)}")
            st.metric("Road Intersections", f"{st.session_state.features.get('road_intersection_count', 0)}")
            st.metric("Green Space Count", f"{st.session_state.features.get('green_space_count', 0)}")
        
        # Display class areas as a chart
        if 'class_areas' in st.session_state.features:
            st.subheader("Class Areas")
            class_areas = st.session_state.features['class_areas']
            
            # Create a bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            classes = list(class_areas.keys())
            percentages = [class_areas[c] for c in classes]
            
            ax.bar(classes, percentages, color=['gray', 'red', 'darkgray', 'green', 'blue'])
            ax.set_ylabel('Percentage (%)')
            ax.set_title('Distribution of Urban Elements')
            ax.set_ylim(0, 100)
            
            # Add percentage labels
            for i, v in enumerate(percentages):
                ax.text(i, v + 1, f"{v:.1f}%", ha='center')
            
            plt.tight_layout()
            chart_path = os.path.join(
                st.session_state.output_dirs["base_dir"],
                f"class_areas_chart_{datetime.now().strftime('%H%M%S')}.png"
            )
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            st.image(chart_path, use_column_width=True)
            
            # Save chart data
            chart_data = {
                "class_areas": class_areas,
                "chart_path": chart_path,
                "timestamp": datetime.now().isoformat()
            }
            
            chart_data_path = save_json_data(
                chart_data,
                st.session_state.output_dirs["overall_outputs"],
                f"class_areas_chart_data_{datetime.now().strftime('%H%M%S')}.json"
            )
        
        # Add download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            # Prepare features for download - convert NumPy types to avoid serialization issues
            features_json = json.dumps(enhanced_convert_numpy_types(st.session_state.features), indent=2)
            st.download_button(
                label="Download Features JSON",
                data=features_json,
                file_name=f"urban_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            # Create CSV for download
            if 'class_areas' in st.session_state.features:
                csv_data = "Feature,Value\n"
                # Add morphological parameters
                for param in ['DN', 'FAR', 'AH', 'HV', 'MV', 'GP', 'WP', 'HP']:
                    csv_data += f"{param},{float(st.session_state.features.get(param, 0)):.6f}\n"
                
                # Add key statistics - convert potential NumPy types to Python native types
                csv_data += f"building_count,{int(st.session_state.features.get('building_count', 0))}\n"
                csv_data += f"road_intersection_count,{int(st.session_state.features.get('road_intersection_count', 0))}\n"
                csv_data += f"green_space_count,{int(st.session_state.features.get('green_space_count', 0))}\n"
                
                st.download_button(
                    label="Download Features CSV",
                    data=csv_data,
                    file_name=f"urban_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Save CSV data
                csv_path = os.path.join(
                    st.session_state.output_dirs["overall_outputs"],
                    f"urban_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                )
                with open(csv_path, 'w') as f:
                    f.write(csv_data)
    
    elif st.session_state.urban_plan is not None and 'morphological_parameters' in st.session_state.urban_plan:
        # Display morphological parameters from the urban plan
        st.subheader("Urban Plan Morphological Parameters")
        
        params = st.session_state.urban_plan['morphological_parameters']
        
        # Create metrics dashboard
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        with metrics_col1:
            st.metric("DN (Density)", f"{params.get('DN', 0):.3f}")
            st.metric("FAR (Floor Area Ratio)", f"{params.get('FAR', 0):.3f}")
            st.metric("AH (Average Height)", f"{params.get('AH', 0):.1f} m")
        
        with metrics_col2:
            st.metric("GP (Green Proportion)", f"{params.get('GP', 0):.3f}")
            st.metric("WP (Water Proportion)", f"{params.get('WP', 0):.3f}")
            st.metric("HP (Hardscape Proportion)", f"{params.get('HP', 0):.3f}")
        
        with metrics_col3:
            st.metric("Total Area", f"{params.get('total_area', 0):.1f} m²")
            st.metric("Building Footprint", f"{params.get('building_footprint_area', 0):.1f} m²")
            st.metric("Floor Area", f"{params.get('floor_area', 0):.1f} m²")
        
        # Download urban plan data
        if 'parameters_output' in st.session_state.urban_plan:
            with open(st.session_state.urban_plan['parameters_output'], 'r') as f:
                params_json = f.read()
                
            st.download_button(
                label="Download Urban Plan Parameters",
                data=params_json,
                file_name=f"urban_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    else:
        st.info("Extract features in the UNet tab or generate an urban plan in the Voronoi Planning tab first")

    # Add UDGAN Output Section
    if st.session_state.urban_plan is not None:
        st.header("UDGAN Output")
        
        # Display the final urban plan and 2D visualization with buildings if available
        if 'visualization_output' in st.session_state.urban_plan and st.session_state.urban_plan['visualization_output']:
            # If we have 2D visualization with buildings, show it
            if st.session_state.urban_2d_outputs and 'files' in st.session_state.urban_2d_outputs and 'vis_2d' in st.session_state.urban_2d_outputs['files']:
                st.image(st.session_state.urban_2d_outputs['files']['vis_2d'], 
                        caption="2D Urban Layout with Buildings", use_column_width=True)
                
                # Add download button for 2D visualization
                with open(st.session_state.urban_2d_outputs['files']['vis_2d'], "rb") as file:
                    st.download_button(
                        label="Download 2D Visualization Image",
                        data=file,
                        file_name=os.path.basename(st.session_state.urban_2d_outputs['files']['vis_2d']),
                        mime="image/png"
                    )
            else:
                # Otherwise show the basic Voronoi plan
                st.image(st.session_state.urban_plan['visualization_output'], 
                        caption="Generated Urban Plan", use_column_width=True)
            
            # Add Export GAN Data button
            if st.button("Export Complete Data for GAN"):
                with st.spinner("Compiling and saving all data for GAN input..."):
                    try:
                        # Create comprehensive dataset for GAN
                        gan_input_data = {
                            "project_id": f"UDGAN_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            "timestamp": datetime.now().isoformat(),
                            "input_image_path": st.session_state.image_path,
                            "is_sketch_input": st.session_state.is_sketch_input,
                            
                            # User constraints
                            "urban_constraints": {
                                "area_scale": area_scale,
                                "layout_shape": layout_shape,
                                "road_network_type": road_network_type,
                                "building_density": building_density,
                                "sustainability_score": sustainability_score,
                                "zoning_distribution": zoning_distribution,
                                "num_zones": num_zones
                            },
                            
                            # UNet features
                            "unet_features": st.session_state.features,
                            
                            # Voronoi data
                            "voronoi_preparation": st.session_state.voronoi_data,
                            
                            # Urban plan
                            "urban_plan_summary": {
                                "morphological_parameters": st.session_state.urban_plan.get('morphological_parameters', {}),
                                "road_network": st.session_state.urban_plan.get('road_network', {}),
                                "zones": {
                                    "count": len(st.session_state.urban_plan.get('regions', [])),
                                    "zone_assignments": st.session_state.urban_plan.get('zone_assignments', [])
                                }
                            },
                            
                            # 2D visualization data if available
                            "urban_2d_outputs": st.session_state.urban_2d_outputs,
                            
                            # Tensor data for GAN
                            "tensor_data": st.session_state.urban_plan.get('tensor_data', {}).get('metadata', {})
                        }
                        
                        # Save comprehensive data
                        gan_data_path = save_json_data(
                            gan_input_data,
                            st.session_state.output_dirs["overall_outputs"],
                            f"gan_input_data_{datetime.now().strftime('%H%M%S')}.json"
                        )
                        
                        st.success(f"Complete GAN input data exported!")
                        
                        # Create download button
                        with open(gan_data_path, 'r') as f:
                            json_data = f.read()
                            
                        st.download_button(
                            label="Download GAN Input JSON",
                            data=json_data,
                            file_name=f"udgan_input_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                        
                        # Add 2D visualization info if available
                        if st.session_state.urban_2d_outputs and 'files' in st.session_state.urban_2d_outputs:
                            st.write("2D urban visualization with buildings is included in the exported data.")
                    except Exception as e:
                        st.error(f"Error exporting GAN data: {e}")
                        traceback.print_exc()
            
            # Download tensor button
            if 'tensor_data' in st.session_state.urban_plan and st.session_state.urban_plan['tensor_data']:
                tensor_path = os.path.join(
                    st.session_state.output_dirs["overall_outputs"],
                    f"udgan_tensor_{datetime.now().strftime('%H%M%S')}.npy"
                )
                np.save(tensor_path, st.session_state.urban_plan['tensor_data']['tensor'])
                st.download_button(
                    label="Download UDGAN Tensor",
                    data=open(tensor_path, 'rb').read(),
                    file_name=f"udgan_tensor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy",
                    mime="application/octet-stream"
                )

# 3D Visualization Section - NEW TAB
with tab4:
    st.header("3D City Visualization")
    
    if st.session_state.urban_plan is not None and 'visualization_output' in st.session_state.urban_plan:
        st.subheader("Generate Full 3D City Model")
        
        if st.button("Generate 3D City Visualization"):
            with st.spinner("Generating full 3D city model - this may take several minutes..."):
                try:
                    # Create output directory for 3D city if needed
                    os.makedirs(os.path.join(st.session_state.output_dirs["base_dir"], "6_3d_city"), exist_ok=True)
                    
                    # Get Voronoi diagram image path from urban plan
                    voronoi_image_path = st.session_state.urban_plan['visualization_output']
                    
                    # Generate 3D city visualization
                    city_3d_outputs = generate_3d_city_visualization(
                        voronoi_image_path,
                        os.path.join(st.session_state.output_dirs["base_dir"], "6_3d_city")
                    )
                    
                    if city_3d_outputs and 'html_output' in city_3d_outputs:
                        st.session_state.city_3d_outputs = city_3d_outputs
                        st.success("3D city visualization generated successfully!")
                        
                        # Display info about the generation
                        st.info(f"Generated {city_3d_outputs['building_count']} buildings and {city_3d_outputs['tree_count']} trees")
                        
                        # Provide link to HTML file
                        html_file_path = city_3d_outputs['html_output']
                        file_name = os.path.basename(html_file_path)
                        
                        # Create download button for HTML file
                        with open(html_file_path, "rb") as file:
                            st.download_button(
                                label="Download 3D City HTML Visualization",
                                data=file,
                                file_name=file_name,
                                mime="text/html"
                            )
                        
                        # Try to display an image preview or embedded HTML
                        st.subheader("3D Visualization Preview")
                        st.components.v1.html(open(html_file_path, 'r').read(), height=600)
                        
                    else:
                        st.warning("Could not generate 3D city visualization. Please check the logs for details.")
                except Exception as e:
                    st.error(f"Error in 3D city generation: {e}")
                    traceback.print_exc()
    elif st.session_state.city_3d_outputs is not None and 'html_output' in st.session_state.city_3d_outputs:
        # Show previously generated 3D visualization
        st.success("3D city visualization was previously generated")
        
        # Display info about the generation
        st.info(f"Generated {st.session_state.city_3d_outputs['building_count']} buildings and {st.session_state.city_3d_outputs['tree_count']} trees")
        
        # Provide link to HTML file
        html_file_path = st.session_state.city_3d_outputs['html_output']
        file_name = os.path.basename(html_file_path)
        
        # Create download button for HTML file
        with open(html_file_path, "rb") as file:
            st.download_button(
                label="Download 3D City HTML Visualization",
                data=file,
                file_name=file_name,
                mime="text/html"
            )
        
        # Display the embedded HTML
        st.subheader("3D Visualization Preview")
        st.components.v1.html(open(html_file_path, 'r').read(), height=600)
        
        # Regenerate button
        if st.button("Regenerate 3D Visualization"):
            st.experimental_rerun()
    else:
        st.info("Please generate an urban plan in the Voronoi Planning tab first")

# Add footer
st.markdown("---")
st.markdown("""
<div style="text-align: center;">
    <p>UDGAN Framework Implementation • Urban Design Generative Adversarial Network</p>
    <p style="font-size: 0.8em;">Built with Streamlit • UNet • Voronoi • 3D Visualization</p>
</div>
""", unsafe_allow_html=True)
