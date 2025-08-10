import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth
from skimage.measure import find_contours
import json
from datetime import datetime
import traceback
from BoundaryProcessor import BoundaryProcessor  # Import the new BoundaryProcessor

class UNetProcessor:
    """
    UNet segmentation processor for the Streamlit interface.
    This class wraps the UNetSegmentation class from Unetclaude.py
    """
    
    def __init__(self, input_size=(256, 256, 3)):
        """Initialize the UNet processor"""
        self.input_size = input_size
        self.class_names = ["Background", "Building", "Road", "Green Space", "Water"]
        self.num_classes = len(self.class_names)
        self.model = None
        self.scale_factor = 1.0  # Default scale (1 pixel = 1 sq ft)
        self.boundary_processor = BoundaryProcessor()  # Initialize the boundary processor
        
        # Dictionary to store colormap for visualization
        self.colormap = {
            0: [150, 150, 150],  # Background (gray)
            1: [230, 25, 75],    # Buildings (red)
            2: [60, 60, 60],     # Roads (dark gray)
            3: [60, 180, 75],    # Green space (green)
            4: [0, 130, 200]     # Water (blue)
        }
        
        # Build the model
        self.build_unet_model()
    
    def set_scale_factor(self, scale_factor):
        """Set the scale factor for area calculations"""
        self.scale_factor = scale_factor
    
    def build_unet_model(self):
        """Build the UNet model architecture"""
        # Input
        inputs = Input(self.input_size)
        
        # Encoder (Contracting) Path
        # Block 1
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        
        # Block 2
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        
        # Block 3
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        
        # Block 4
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
        
        # Bridge
        conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
        drop5 = Dropout(0.5)(conv5)
        
        # Decoder (Expanding) Path
        # Block 6
        up6 = Conv2D(512, 2, activation='relu', padding='same')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)
        
        # Block 7
        up7 = Conv2D(256, 2, activation='relu', padding='same')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)
        
        # Block 8
        up8 = Conv2D(128, 2, activation='relu', padding='same')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)
        
        # Block 9
        up9 = Conv2D(64, 2, activation='relu', padding='same')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
        
        # Output layer - multi-channel segmentation masks
        outputs = Conv2D(self.num_classes, 1, activation='softmax')(conv9)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(optimizer=Adam(learning_rate=1e-4), 
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        self.model = model
        return model
    
    def load_model(self, model_path):
        """Load a pre-trained UNet model"""
        try:
            self.model = load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def preprocess_image(self, image_path):
        """Preprocess image for segmentation"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to read image: {image_path}")
                
            # Convert to RGB (from BGR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to input dimensions
            img_resized = cv2.resize(img_rgb, (self.input_size[1], self.input_size[0]))
            
            # Normalize pixel values to [0, 1]
            img_normalized = img_resized / 255.0
            
            return {
                "original": img_rgb,
                "resized": img_resized,
                "normalized": img_normalized,
                "path": image_path
            }
            
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            traceback.print_exc()
            return None
    
    def segment_image(self, image_path):
        """Perform semantic segmentation on input image"""
        # Preprocess image
        img_data = self.preprocess_image(image_path)
        if img_data is None:
            return None
        
        # First, check if this is a sketch/hand-drawn boundary using BoundaryProcessor
        if self.boundary_processor.is_sketch_input(img_data["original"]):
            print("Detected sketch input. Using specialized boundary extraction.")
            boundary_extraction = self.boundary_processor.extract_boundary_from_sketch(img_data["original"])
            
            if boundary_extraction and len(boundary_extraction["boundary"]) >= 3:
                print("Successfully extracted boundary from sketch.")
                # Create a mock segmentation result with the extracted boundary
                h, w = self.input_size[0], self.input_size[1]
                
                # Create empty segmentation mask
                segmentation_mask = np.zeros((h, w), dtype=np.uint8)
                
                # Create boundary mask
                boundary_mask = np.zeros((h, w), dtype=np.uint8)
                
                # Scale boundary points to match the resized dimensions
                orig_h, orig_w = img_data["original"].shape[:2]
                scaled_boundary = []
                for x, y in boundary_extraction["boundary"]:
                    scaled_x = int(x * w / orig_w)
                    scaled_y = int(y * h / orig_h)
                    scaled_boundary.append([scaled_x, scaled_y])
                
                # Draw the boundary polygon
                boundary_points = np.array([scaled_boundary], dtype=np.int32)
                cv2.fillPoly(boundary_mask, boundary_points, 255)
                
                # Set everything inside the boundary as "Background"
                segmentation_mask[boundary_mask > 0] = 0
                
                # Create edge detection
                edges = cv2.Canny(boundary_mask, 50, 150)
                
                # Create one-hot encoded prediction
                prediction = np.zeros((h, w, self.num_classes), dtype=np.float32)
                for i in range(self.num_classes):
                    prediction[:, :, i] = (segmentation_mask == i).astype(np.float32)
                
                # Add results to the output dictionary
                img_data["prediction"] = prediction
                img_data["segmentation_mask"] = segmentation_mask
                img_data["edges"] = edges
                img_data["boundary"] = np.array(scaled_boundary)
                img_data["is_sketch"] = True
                img_data["boundary_extraction"] = boundary_extraction
                
                return img_data
        
        # If not a sketch or boundary extraction failed, proceed with normal UNet segmentation
        # If model isn't trained properly, use mock segmentation
        if not hasattr(self.model, 'optimizer') or self.model.optimizer is None:
            print("Using mock segmentation (model not fully trained)")
            return self._create_mock_segmentation(img_data)
        
        # Perform prediction
        try:
            # Add batch dimension and predict
            x = np.expand_dims(img_data["normalized"], axis=0)
            prediction = self.model.predict(x)
            
            # Remove batch dimension
            prediction = prediction[0]
            
            # Get class with highest probability for each pixel
            segmentation_mask = np.argmax(prediction, axis=-1)
            
            # Create grayscale edge map using Canny edge detection
            gray = cv2.cvtColor(img_data["resized"], cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            img_data["edges"] = edges
            
            # Extract boundary
            boundary = self._extract_boundary(segmentation_mask)
            
            # Add results to output dictionary
            img_data["prediction"] = prediction
            img_data["segmentation_mask"] = segmentation_mask
            img_data["boundary"] = boundary
            img_data["is_sketch"] = False
            
            return img_data
            
        except Exception as e:
            print(f"Error performing segmentation: {str(e)}")
            traceback.print_exc()
            return None
    
    def _create_mock_segmentation(self, img_data):
        """Create mock segmentation for demonstration purposes"""
        h, w = self.input_size[0], self.input_size[1]
        
        # Create mock segmentation based on color thresholds
        rgb = img_data["resized"]
        
        # Simple color-based segmentation for demonstration
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Define thresholds for each class based on RGB values
        # Buildings (detect red and brown tones)
        building_mask = ((rgb[:,:,0] > 100) & (rgb[:,:,1] < 100) & (rgb[:,:,2] < 100)) | \
                        ((rgb[:,:,0] > 140) & (rgb[:,:,1] > 80) & (rgb[:,:,1] < 140) & (rgb[:,:,2] < 100))
        mask[building_mask] = 1
        
        # Roads (detect dark gray and black)
        road_mask = (rgb[:,:,0] < 100) & (rgb[:,:,1] < 100) & (rgb[:,:,2] < 100) & \
                    ~((rgb[:,:,0] < 30) & (rgb[:,:,1] > 30) & (rgb[:,:,2] > 100))  # Exclude water
        mask[road_mask] = 2
        
        # Green spaces (detect greens)
        green_mask = (rgb[:,:,1] > 100) & (rgb[:,:,0] < rgb[:,:,1]) & (rgb[:,:,2] < rgb[:,:,1])
        mask[green_mask] = 3
        
        # Water (detect blues)
        water_mask = (rgb[:,:,2] > 130) & (rgb[:,:,0] < 100) & (rgb[:,:,1] < 130)
        mask[water_mask] = 4
        
        # Create mock prediction with one-hot encoding
        prediction = np.zeros((h, w, self.num_classes))
        for i in range(self.num_classes):
            prediction[:,:,i] = (mask == i).astype(np.float32)
        
        # Add edge detection
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Check if this might be a sketch - if so, use the boundary processor
        if self.boundary_processor.is_sketch_input(img_data["original"]):
            print("Mock segmentation detected sketch input. Using specialized boundary extraction.")
            boundary_extraction = self.boundary_processor.extract_boundary_from_sketch(img_data["original"])
            
            if boundary_extraction and len(boundary_extraction["boundary"]) >= 3:
                # Scale boundary points to match the resized dimensions
                orig_h, orig_w = img_data["original"].shape[:2]
                scaled_boundary = []
                for x, y in boundary_extraction["boundary"]:
                    scaled_x = int(x * w / orig_w)
                    scaled_y = int(y * h / orig_h)
                    scaled_boundary.append([scaled_x, scaled_y])
                
                boundary = np.array(scaled_boundary)
                img_data["is_sketch"] = True
                img_data["boundary_extraction"] = boundary_extraction
            else:
                # Extract boundary with standard method
                boundary = self._extract_boundary(mask)
                img_data["is_sketch"] = False
        else:
            # Extract boundary with standard method
            boundary = self._extract_boundary(mask)
            img_data["is_sketch"] = False
        
        # Add results to output dictionary
        img_data["prediction"] = prediction
        img_data["segmentation_mask"] = mask
        img_data["edges"] = edges
        img_data["boundary"] = boundary
        
        return img_data
    
    def visualize_segmentation(self, segmentation_data, output_file=None):
        """Visualize segmentation or boundary extraction results"""
        if segmentation_data is None:
            print("No segmentation data to visualize.")
            return None
        
        # Check if this was a sketch input with boundary extraction
        if segmentation_data.get("is_sketch", False) and "boundary_extraction" in segmentation_data:
            # Use boundary_processor to visualize the extraction process
            vis_path = self.boundary_processor.visualize_boundary_extraction(
                segmentation_data["original"],
                segmentation_data["boundary_extraction"],
                output_file
            )
            return vis_path
        
        # Standard visualization for normal segmentation
        if "segmentation_mask" not in segmentation_data:
            print("No segmentation mask found in data.")
            return None
        
        # Create RGB visualization of segmentation mask
        h, w = segmentation_data["segmentation_mask"].shape
        vis_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Apply colormap
        for class_id, color in self.colormap.items():
            vis_img[segmentation_data["segmentation_mask"] == class_id] = color
        
        # Create 2x2 subplot figure
        plt.figure(figsize=(12, 10))
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axs[0, 0].imshow(segmentation_data["original"])
        axs[0, 0].set_title("Original Image")
        axs[0, 0].axis("off")
        
        # Segmentation mask
        axs[0, 1].imshow(vis_img)
        axs[0, 1].set_title("Segmentation")
        axs[0, 1].axis("off")
        
        # Edge detection
        axs[1, 0].imshow(segmentation_data["edges"], cmap="gray")
        axs[1, 0].set_title("Edge Detection")
        axs[1, 0].axis("off")
        
        # Boundary overlay
        overlay = segmentation_data["resized"].copy()
        if segmentation_data["boundary"] is not None and len(segmentation_data["boundary"]) > 0:
            # Draw boundary on copy of original image
            boundary = np.array([segmentation_data["boundary"]], dtype=np.int32)
            cv2.polylines(overlay, boundary, True, (255, 0, 0), 2)
        
        axs[1, 1].imshow(overlay)
        axs[1, 1].set_title("Boundary Detection")
        axs[1, 1].axis("off")
        
        # Add legend for segmentation colors
        legend_patches = []
        for i, class_name in enumerate(self.class_names):
            color = self.colormap.get(i, [0, 0, 0])
            color_normalized = [c/255 for c in color]
            patch = plt.Rectangle((0, 0), 1, 1, fc=color_normalized)
            legend_patches.append(patch)
        
        fig.legend(legend_patches, self.class_names, loc="lower center", ncol=len(self.class_names))
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        
        # Save figure if output path is provided
        if output_file:
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close(fig)
            plt.close()
            return output_file
        else:
            plt.show()
            plt.close(fig)
            plt.close()
            return None
    
    def _extract_boundary(self, segmentation_mask):
        """Extract boundary from segmentation mask with improved error handling and timeout"""
        try:
            # Create binary mask (foreground vs background)
            binary = np.zeros_like(segmentation_mask, dtype=np.uint8)
            binary[segmentation_mask > 0] = 255  # All non-background as foreground
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # Set processing timeout
            start_time = time.time()
            max_processing_time = 30  # seconds
            
            # Try find_contours for more accurate boundary detections
            try:
                # Check timeout
                if time.time() - start_time > max_processing_time:
                    print("Timeout in find_contours processing")
                    raise TimeoutError("Boundary extraction timeout")
                    
                contours = find_contours(binary, 0.5)
                
                if contours and len(contours) > 0:
                    # Find the largest contour by area
                    largest_contour = max(contours, key=lambda x: len(x))
                    
                    # Simplify contour while preserving shape details
                    # Convert to the format expected by cv2.approxPolyDP
                    largest_contour_int = np.array(largest_contour, dtype=np.float32)
                    # Flip x,y coordinates to match OpenCV convention
                    largest_contour_int = np.fliplr(largest_contour_int)
                    
                    # Reshape for approxPolyDP
                    largest_contour_reshaped = largest_contour_int.reshape((-1, 1, 2))
                    
                    # Use adaptive epsilon based on perimeter
                    perimeter = cv2.arcLength(largest_contour_reshaped, True)
                    epsilon = 0.005 * perimeter  # Use a small epsilon to preserve detail
                    approx = cv2.approxPolyDP(largest_contour_reshaped, epsilon, True)
                    
                    # Return as numpy array
                    simplified = approx.squeeze()
                    
                    # Ensure we have at least 3 points
                    if len(simplified) >= 3:
                        return simplified
            except Exception as e:
                print(f"Error with skimage contours: {e}")
                # Fallback to OpenCV's findContours if skimage method fails
            
            # Check timeout
            if time.time() - start_time > max_processing_time:
                print("Timeout in boundary extraction")
                raise TimeoutError("Boundary extraction timeout")
                
            # Traditional OpenCV contour finding as fallback
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Use a smaller epsilon for more detailed boundary
                epsilon = 0.005 * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                # Ensure we have at least 3 points
                if approx.shape[0] >= 3:
                    return approx.squeeze()
            
            print("No valid boundary found")
            return np.array([])
        except TimeoutError as e:
            print(f"Timeout in boundary extraction: {e}")
            return np.array([])
        except Exception as e:
            print(f"Error in boundary extraction: {str(e)}")
            traceback.print_exc()
            # Fallback to simplest method if all else fails
            try:
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    return contours[0].squeeze()
            except:
                pass
            return np.array([])
    
    def extract_features(self, segmentation_data):
        """Extract urban features from segmentation results"""
        if segmentation_data is None or "segmentation_mask" not in segmentation_data:
            print("No segmentation data for feature extraction.")
            return None
        
        # Initialize features dictionary
        features = {}
        
        mask = segmentation_data["segmentation_mask"]
        h, w = mask.shape
        total_pixels = h * w
        
        # Extract area percentages for each class
        class_areas = {}
        for i, class_name in enumerate(self.class_names):
            class_pixels = np.sum(mask == i)
            class_percentage = (class_pixels / total_pixels) * 100
            class_areas[class_name] = class_percentage
        
        features["class_areas"] = class_areas
        
        # Calculate total area in square feet using the scale factor
        total_area_sq_ft = total_pixels * self.scale_factor
        features["total_area_sq_ft"] = total_area_sq_ft
        
        # Calculate area of the boundary polygon
        if "boundary" in segmentation_data and segmentation_data["boundary"] is not None and len(segmentation_data["boundary"]) > 2:
            # Calculate area of polygon using shoelace formula
            x = segmentation_data["boundary"][:, 0]
            y = segmentation_data["boundary"][:, 1]
            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            features["boundary_area_pixels"] = area
            features["boundary_area_sq_ft"] = area * self.scale_factor
            
            # Calculate perimeter
            perimeter = 0
            for i in range(len(segmentation_data["boundary"])):
                p1 = segmentation_data["boundary"][i]
                p2 = segmentation_data["boundary"][(i + 1) % len(segmentation_data["boundary"])]
                perimeter += np.sqrt(((p2[0] - p1[0]) ** 2) + ((p2[1] - p1[1]) ** 2))
            features["boundary_perimeter_pixels"] = perimeter
            features["boundary_perimeter_ft"] = perimeter * np.sqrt(self.scale_factor)
        
        # Building-specific features
        building_mask = (mask == 1).astype(np.uint8) * 255
        
        # Count building instances
        building_contours, _ = cv2.findContours(building_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        features["building_count"] = len(building_contours)
        
        # Building size statistics
        if building_contours:
            building_areas = [cv2.contourArea(contour) for contour in building_contours]
            features["avg_building_size"] = np.mean(building_areas)
            features["min_building_size"] = np.min(building_areas)
            features["max_building_size"] = np.max(building_areas)
            features["building_size_variance"] = np.var(building_areas)
        else:
            features["avg_building_size"] = 0
            features["min_building_size"] = 0
            features["max_building_size"] = 0
            features["building_size_variance"] = 0
        
        # Road network features
        road_mask = (mask == 2).astype(np.uint8) * 255
        
        # Skeletonize roads to get network
        try:
            if hasattr(cv2, 'ximgproc'):
                road_skeleton = cv2.ximgproc.thinning(road_mask)
            else:
                # Fallback to simpler skeletonization
                from skimage.morphology import skeletonize
                road_skeleton = skeletonize(road_mask > 0).astype(np.uint8) * 255
            
            # Count road intersections
            road_intersections = self._detect_intersections(road_skeleton)
            features["road_intersection_count"] = len(road_intersections)
            
            # Road length (approximation)
            features["road_length"] = np.sum(road_skeleton > 0)
        except Exception as e:
            print(f"Error processing road features: {e}")
            traceback.print_exc()
            features["road_intersection_count"] = 0
            features["road_length"] = 0
        
        # Road-to-area ratio
        features["road_area_ratio"] = class_areas.get("Road", 0) / 100
        
        # Green space features
        green_mask = (mask == 3).astype(np.uint8) * 255
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        features["green_space_count"] = len(green_contours)
        
        if green_contours:
            green_areas = [cv2.contourArea(contour) for contour in green_contours]
            features["avg_green_space_size"] = np.mean(green_areas)
            features["green_space_coverage"] = class_areas.get("Green Space", 0) / 100
        else:
            features["avg_green_space_size"] = 0
            features["green_space_coverage"] = 0
        
        # Water features
        water_mask = (mask == 4).astype(np.uint8) * 255
        water_contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        features["water_body_count"] = len(water_contours)
        features["water_coverage"] = class_areas.get("Water", 0) / 100
        
        # Morphological parameters as defined in the UDGAN framework
        features["DN"] = class_areas.get("Building", 0) / 100  # Density factor
        # Estimate FAR (Floor Area Ratio) - using a default average of 3 floors per building
        features["FAR"] = features["DN"] * 3
        # Assuming average height is proportional to building area 
        features["AH"] = 10.0  # Placeholder average height in meters
        features["HV"] = features["building_size_variance"] / 1000 if features["building_size_variance"] > 0 else 0  # Height variance
        features["MV"] = features["building_size_variance"] / features["avg_building_size"] if features["avg_building_size"] > 0 else 0  # Building massing variance
        features["GP"] = features["green_space_coverage"]  # Green area proportion
        features["WP"] = features["water_coverage"]  # Water surface proportion
        features["HP"] = class_areas.get("Road", 0) / 100  # Hardscape proportion
        
        # Add metadata
        features["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        features["image_path"] = segmentation_data.get("path", "Unknown")
        features["image_dimensions"] = f"{w}x{h} pixels"
        features["scale_factor"] = self.scale_factor
        features["scale_units"] = "square feet per pixel"
        
        return features
    
    def _detect_intersections(self, skeleton_image):
        """Detect intersections in road network skeleton"""
        # Create a copy of the skeleton
        skel = skeleton_image.copy()
        
        # Initialize list to store intersections
        intersections = []
        
        # Define kernel for neighbor counting
        neighbor_kernel = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ], dtype=np.uint8)
        
        try:
            # Count neighbors for each pixel
            neighbors = cv2.filter2D(skel, -1, neighbor_kernel)
            
            # Intersections have 3 or more neighbors
            intersection_mask = (neighbors >= 3) & (skel > 0)
            
            # Get intersection coordinates
            y, x = np.where(intersection_mask > 0)
            intersections = list(zip(x, y))
            
            # Cluster nearby intersections if we have enough points
            if len(intersections) > 1:
                try:
                    # Use Mean Shift clustering for better clustering of nearby points
                    points = np.array(intersections)
                    
                    # Estimate bandwidth automatically
                    bandwidth = estimate_bandwidth(points, quantile=0.2, n_samples=min(len(points), 500))
                    if bandwidth <= 0:
                        bandwidth = 5  # Default fallback
                    
                    # Apply MeanShift clustering
                    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
                    ms.fit(points)
                    clustered_centers = ms.cluster_centers_
                    
                    # Convert to integer coordinates
                    clustered_intersections = [(int(center[0]), int(center[1])) for center in clustered_centers]
                    return clustered_intersections
                except Exception as e:
                    print(f"Error in MeanShift clustering: {str(e)}")
                    # Fallback to DBSCAN
                    try:
                        clustering = DBSCAN(eps=5, min_samples=1).fit(intersections)
                        labels = clustering.labels_
                        
                        # Get cluster centers
                        clustered_intersections = []
                        for label in np.unique(labels):
                            indices = np.where(labels == label)[0]
                            points = np.array([intersections[i] for i in indices])
                            center = np.mean(points, axis=0).astype(int)
                            clustered_intersections.append((center[0], center[1]))
                        
                        return clustered_intersections
                    except Exception as e:
                        print(f"Error in DBSCAN clustering: {str(e)}")
                        # Return raw intersections as fallback
                        return intersections
            else:
                return intersections
        except Exception as e:
            print(f"Error in intersection detection: {str(e)}")
            traceback.print_exc()
            # Fallback to simple method
            try:
                _, _, stats, centroids = cv2.connectedComponentsWithStats(skel, connectivity=8)
                junctions = []
                for i in range(1, len(stats)):
                    if stats[i, cv2.CC_STAT_AREA] > 0:
                        junctions.append((int(centroids[i, 0]), int(centroids[i, 1])))
                return junctions
            except:
                return []
    
    def create_binary_masks(self, segmentation_data):
        """Create binary masks for each segmented element"""
        if segmentation_data is None or "segmentation_mask" not in segmentation_data:
            print("No segmentation data for creating binary masks.")
            return None
        
        mask = segmentation_data["segmentation_mask"]
        
        # Create binary masks for each class
        binary_masks = {}
        for i, class_name in enumerate(self.class_names):
            # Create binary mask
            binary = np.zeros_like(mask, dtype=np.uint8)
            binary[mask == i] = 255
            
            # Store in dictionary
            binary_masks[class_name.lower().replace(" ", "_")] = binary
        
        return binary_masks
    
    def create_tensor_representation(self, segmentation_data):
        """Create tensor representation for UDGAN input"""
        if segmentation_data is None or "segmentation_mask" not in segmentation_data:
            print("No segmentation data for tensor creation.")
            return None
        
        mask = segmentation_data["segmentation_mask"]
        h, w = mask.shape
        
        # Create 4-channel tensor as specified in the framework document
        # Channel 0: Buildings
        # Channel 1: Roads
        # Channel 2: Green spaces
        # Channel 3: Water bodies
        tensor = np.zeros((h, w, 4), dtype=np.float32)
        
        # Building channel
        tensor[:, :, 0] = (mask == 1).astype(np.float32)
        
        # Road channel
        tensor[:, :, 1] = (mask == 2).astype(np.float32)
        
        # Green space channel
        tensor[:, :, 2] = (mask == 3).astype(np.float32)
        
        # Water channel
        tensor[:, :, 3] = (mask == 4).astype(np.float32)
        
        return tensor
    
    def prepare_for_voronoi(self, segmentation_data):
        """Prepare data for Voronoi diagram generation"""
        if segmentation_data is None:
            print("No segmentation data for Voronoi preparation.")
            return None
        
        # For sketch inputs, use the extracted boundary directly
        if segmentation_data.get("is_sketch", False) and "boundary" in segmentation_data:
            boundary = segmentation_data["boundary"]
            
            # Verify boundary has enough points and is valid
            if len(boundary) < 3:
                print("Invalid boundary: needs at least 3 points")
                return None
                
            try:
                # Verify it's a valid polygon
                from shapely.geometry import Polygon
                poly = Polygon(boundary)
                if not poly.is_valid:
                    # Try to fix invalid polygon
                    poly = poly.buffer(0)
                    if not poly.is_valid:
                        print("Cannot create valid polygon from boundary")
                        return None
                    # Update boundary if needed
                    boundary = np.array(poly.exterior.coords)[:-1]
            except Exception as e:
                print(f"Error validating boundary from sketch: {e}")
                traceback.print_exc()
                boundary = self._extract_boundary(segmentation_data.get("segmentation_mask", None))
        else:
            # Extract boundary from segmentation mask
            boundary = self._extract_boundary(segmentation_data.get("segmentation_mask", None))
        
        # Extract building centroids if available
        building_centroids = []
        if not segmentation_data.get("is_sketch", False) and "segmentation_mask" in segmentation_data:
            mask = segmentation_data["segmentation_mask"]
            building_mask = (mask == 1).astype(np.uint8) * 255
            
            # Apply distance transform to find the center of each building
            dist_transform = cv2.distanceTransform(building_mask, cv2.DIST_L2, 5)
            
            # Apply threshold to get foreground areas
            ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0) if dist_transform.max() > 0 else (None, np.zeros_like(building_mask))
            sure_fg = sure_fg.astype(np.uint8)
            
            # Label connected components
            num_labels, labels = cv2.connectedComponents(sure_fg)
            
            # Find centroids of each connected component
            for i in range(1, num_labels):
                # Create mask for each building
                building_component = (labels == i).astype(np.uint8) * 255
                # Find contour
                contours, _ = cv2.findContours(building_component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Calculate area of the building
                    area = cv2.contourArea(contours[0])
                    if area > 50:  # Ignore very small components (noise)
                        # Get centroid
                        M = cv2.moments(contours[0])
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            building_centroids.append((cx, cy))
        
        # Get road intersections if available
        road_intersections = []
        if not segmentation_data.get("is_sketch", False) and "segmentation_mask" in segmentation_data:
            mask = segmentation_data["segmentation_mask"]
            road_mask = (mask == 2).astype(np.uint8) * 255
            
            # Better road skeletonization with preprocessing
            # First, ensure roads have good connectivity
            kernel = np.ones((3, 3), np.uint8)
            road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
            
            # Then skeletonize
            road_skeleton = None
            try:
                if hasattr(cv2, 'ximgproc'):
                    road_skeleton = cv2.ximgproc.thinning(road_mask)
                else:
                    # Fallback to simpler skeletonization
                    from skimage.morphology import skeletonize
                    road_skeleton = skeletonize(road_mask > 0).astype(np.uint8) * 255
                    
                # Detect intersections
                road_intersections = self._detect_intersections(road_skeleton)
            except Exception as e:
                print(f"Error in road skeleton processing: {e}")
                road_intersections = []
        
        # Extract features for additional context
        features = self.extract_features(segmentation_data)
        
        # Add distance measurements along boundary segments (for visualization)
        boundary_with_measurements = []
        if boundary is not None and len(boundary) > 2:
            for i in range(len(boundary)):
                pt1 = boundary[i]
                pt2 = boundary[(i + 1) % len(boundary)]
                dist = np.sqrt(np.sum((pt2 - pt1) ** 2))
                # Convert to feet
                dist_feet = dist * np.sqrt(self.scale_factor)
                # Save the midpoint and distance
                mid_point = (pt1 + pt2) / 2
                boundary_with_measurements.append({
                    'start': pt1.tolist() if isinstance(pt1, np.ndarray) else pt1,
                    'end': pt2.tolist() if isinstance(pt2, np.ndarray) else pt2,
                    'midpoint': mid_point.tolist() if isinstance(mid_point, np.ndarray) else mid_point,
                    'distance': float(dist_feet)
                })
        
        return {
            "boundary": boundary,
            "building_centroids": building_centroids,
            "road_intersections": road_intersections,
            "features": features,
            "boundary_measurements": boundary_with_measurements
        }
    
    def visualize_voronoi_preparation(self, voronoi_data, original_image, output_path=None):
        """Visualize Voronoi preparation data"""
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Show original image as background
        if original_image is not None:
            ax.imshow(original_image)
        
        # Plot boundary
        if voronoi_data["boundary"] is not None and len(voronoi_data["boundary"]) > 0:
            boundary = np.vstack([voronoi_data["boundary"], voronoi_data["boundary"][0]])  # Close the boundary
            ax.plot(boundary[:, 0], boundary[:, 1], 'r-', linewidth=2, label="Boundary")
            
            # Add distance measurements if available
            if "boundary_measurements" in voronoi_data and voronoi_data["boundary_measurements"]:
                for measure in voronoi_data["boundary_measurements"]:
                    midpoint = measure["midpoint"]
                    distance = measure["distance"]
                    ax.annotate(f"{int(distance)} ft", xy=(midpoint[0], midpoint[1]), 
                               xytext=(midpoint[0] + 10, midpoint[1] + 10),
                               arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        
        # Plot building centroids
        if voronoi_data["building_centroids"] and len(voronoi_data["building_centroids"]) > 0:
            centroids = np.array(voronoi_data["building_centroids"])
            ax.scatter(centroids[:, 0], centroids[:, 1], c='blue', s=30, marker='o', label="Building Centroids")
        
        # Plot road intersections
        if voronoi_data["road_intersections"] and len(voronoi_data["road_intersections"]) > 0:
            intersections = np.array(voronoi_data["road_intersections"])
            ax.scatter(intersections[:, 0], intersections[:, 1], c='green', s=40, marker='x', label="Road Intersections")
        
        ax.set_title("Voronoi Preparation Data")
        ax.legend()
        
        # Save the visualization
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return output_path
        else:
            plt.show()
            plt.close(fig)
            return None
    
    def save_features_to_json(self, features, output_file):
        """Save extracted features to a JSON file"""
        if features is None:
            print("No features to save.")
            return None
        
        # Function to convert numpy types to Python native types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        try:
            # Convert numpy types to native Python types
            features_copy = convert_types(features)
            
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(features_copy, f, indent=2)
            print(f"Features saved to: {output_file}")
            return output_file
        except Exception as e:
            print(f"Error saving features to JSON: {str(e)}")
            traceback.print_exc()
            return None
        
    # Add this function to the UNetProcessor class in Unet_extend.py

def prepare_for_voronoi(self, segmentation_data):
    """Prepare data for Voronoi diagram generation"""
    if segmentation_data is None:
        print("No segmentation data for Voronoi preparation.")
        return None
    
    # For sketch inputs, use the extracted boundary directly
    if segmentation_data.get("is_sketch", False) and "boundary" in segmentation_data:
        boundary = segmentation_data["boundary"]
        
        # Verify boundary has enough points and is valid
        if len(boundary) < 3:
            print("Invalid boundary: needs at least 3 points")
            return None
            
        try:
            # Verify it's a valid polygon
            from shapely.geometry import Polygon, MultiPolygon
            poly = Polygon(boundary)
            if not poly.is_valid:
                # Try to fix invalid polygon
                poly = poly.buffer(0)
                if not poly.is_valid:
                    print("Cannot create valid polygon from boundary")
                    return None
                
                # Check and handle MultiPolygon case
                if isinstance(poly, MultiPolygon):
                    print("Fixed polygon resulted in a MultiPolygon. Selecting largest polygon.")
                    # Select the largest polygon in the MultiPolygon
                    largest_poly = max(poly.geoms, key=lambda x: x.area)
                    boundary = np.array(largest_poly.exterior.coords)[:-1]
                else:
                    # Update boundary if needed
                    boundary = np.array(poly.exterior.coords)[:-1]
            
        except Exception as e:
            print(f"Error validating boundary from sketch: {e}")
            traceback.print_exc()
            boundary = self._extract_boundary(segmentation_data.get("segmentation_mask", None))
    else:
        # Extract boundary from segmentation mask
        boundary = self._extract_boundary(segmentation_data.get("segmentation_mask", None))
    
    # Extract building centroids if available
    building_centroids = []
    if not segmentation_data.get("is_sketch", False) and "segmentation_mask" in segmentation_data:
        mask = segmentation_data["segmentation_mask"]
        building_mask = (mask == 1).astype(np.uint8) * 255
        
        # Apply distance transform to find the center of each building
        dist_transform = cv2.distanceTransform(building_mask, cv2.DIST_L2, 5)
        
        # Apply threshold to get foreground areas
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0) if dist_transform.max() > 0 else (None, np.zeros_like(building_mask))
        sure_fg = sure_fg.astype(np.uint8)
        
        # Label connected components
        num_labels, labels = cv2.connectedComponents(sure_fg)
        
        # Find centroids of each connected component
        for i in range(1, num_labels):
            # Create mask for each building
            building_component = (labels == i).astype(np.uint8) * 255
            # Find contour
            contours, _ = cv2.findContours(building_component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Calculate area of the building
                area = cv2.contourArea(contours[0])
                if area > 50:  # Ignore very small components (noise)
                    # Get centroid
                    M = cv2.moments(contours[0])
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        building_centroids.append((cx, cy))
    
    # Get road intersections if available
    road_intersections = []
    if not segmentation_data.get("is_sketch", False) and "segmentation_mask" in segmentation_data:
        mask = segmentation_data["segmentation_mask"]
        road_mask = (mask == 2).astype(np.uint8) * 255
        
        # Better road skeletonization with preprocessing
        # First, ensure roads have good connectivity
        kernel = np.ones((3, 3), np.uint8)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
        
        # Then skeletonize
        road_skeleton = None
        try:
            if hasattr(cv2, 'ximgproc'):
                road_skeleton = cv2.ximgproc.thinning(road_mask)
            else:
                # Fallback to simpler skeletonization
                from skimage.morphology import skeletonize
                road_skeleton = skeletonize(road_mask > 0).astype(np.uint8) * 255
                
            # Detect intersections
            road_intersections = self._detect_intersections(road_skeleton)
        except Exception as e:
            print(f"Error in road skeleton processing: {e}")
            road_intersections = []
    
    # Extract features for additional context
    features = self.extract_features(segmentation_data)
    
    # Add distance measurements along boundary segments (for visualization)
    boundary_with_measurements = []
    if boundary is not None and len(boundary) > 2:
        for i in range(len(boundary)):
            pt1 = boundary[i]
            pt2 = boundary[(i + 1) % len(boundary)]
            dist = np.sqrt(np.sum((pt2 - pt1) ** 2))
            # Convert to feet
            dist_feet = dist * np.sqrt(self.scale_factor)
            # Save the midpoint and distance
            mid_point = (pt1 + pt2) / 2
            boundary_with_measurements.append({
                'start': pt1.tolist() if isinstance(pt1, np.ndarray) else pt1,
                'end': pt2.tolist() if isinstance(pt2, np.ndarray) else pt2,
                'midpoint': mid_point.tolist() if isinstance(mid_point, np.ndarray) else mid_point,
                'distance': float(dist_feet)
            })
    
    return {
        "boundary": boundary,
        "building_centroids": building_centroids,
        "road_intersections": road_intersections,
        "features": features,
        "boundary_measurements": boundary_with_measurements
    }
            
    def generate_sample_data(self, num_samples=1, output_dir="sample_data"):
        """Generate sample data for demonstration"""
        os.makedirs(output_dir, exist_ok=True)
        
        samples = {}
        for i in range(num_samples):
            # Create mock urban layout
            h, w = self.input_size[0], self.input_size[1]
            
            # Create random image with urban-like patterns
            img = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Background (light gray)
            img.fill(200)
            
            # Add buildings (red/brown)
            num_buildings = np.random.randint(5, 20)
            for _ in range(num_buildings):
                x = np.random.randint(0, w)
                y = np.random.randint(0, h)
                size_x = np.random.randint(10, 50)
                size_y = np.random.randint(10, 50)
                color = (np.random.randint(150, 200), np.random.randint(50, 100), np.random.randint(50, 100))
                cv2.rectangle(img, (x, y), (x + size_x, y + size_y), color, -1)
            
            # Add roads (dark gray)
            num_roads = np.random.randint(3, 8)
            for _ in range(num_roads):
                x1 = np.random.randint(0, w)
                y1 = np.random.randint(0, h)
                x2 = np.random.randint(0, w)
                y2 = np.random.randint(0, h)
                cv2.line(img, (x1, y1), (x2, y2), (50, 50, 50), np.random.randint(3, 10))
            
            # Add green spaces (green)
            num_green = np.random.randint(2, 6)
            for _ in range(num_green):
                x = np.random.randint(0, w)
                y = np.random.randint(0, h)
                radius = np.random.randint(10, 40)
                color = (np.random.randint(50, 100), np.random.randint(150, 200), np.random.randint(50, 100))
                cv2.circle(img, (x, y), radius, color, -1)
            
            # Add water (blue)
            num_water = np.random.randint(0, 3)
            for _ in range(num_water):
                x = np.random.randint(0, w)
                y = np.random.randint(0, h)
                size_x = np.random.randint(20, 80)
                size_y = np.random.randint(20, 80)
                color = (np.random.randint(50, 100), np.random.randint(100, 150), np.random.randint(150, 250))
                
                # Create irregular water shape
                points = []
                for angle in range(0, 360, 30):
                    dist = size_x * 0.5 * (0.8 + 0.4 * np.random.random())
                    px = x + int(dist * np.cos(np.radians(angle)))
                    py = y + int(dist * np.sin(np.radians(angle)))
                    points.append([px, py])
                
                points = np.array([points], dtype=np.int32)
                cv2.fillPoly(img, points, color)
            
            # Save image
            img_path = os.path.join(output_dir, f"sample_{i}.png")
            cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            samples[f"sample_{i}"] = img_path
        
        return samples