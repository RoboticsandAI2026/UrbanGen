import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import plotly.graph_objects as go
import random
import os
import gc  # For garbage collection
from sklearn.cluster import KMeans
from shapely.geometry import Polygon, Point, box, LineString, MultiPolygon
from shapely.ops import unary_union
from scipy.spatial import Delaunay, Voronoi
import networkx as nx
from shapely.affinity import translate, rotate
import plotly.io as pio
import time  # For performance timing

# Set lower precision to reduce memory usage
np.random.seed(42)  # For reproducibility
FLOAT_PRECISION = np.float32  # Use 32-bit instead of 64-bit floats

# ===== UTILITY FUNCTIONS =====

# Function to load and process the urban plan image - with visualization toggle
def load_and_process_image(filename, show_viz=False):
    print(f"Loading Voronoi diagram: {filename}")
    img = cv2.imread(filename)
    if img is None:
        raise ValueError(f"Could not load image: {filename}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if show_viz:
        plt.figure(figsize=(10, 10))
        plt.imshow(img_rgb)
        plt.title("Voronoi Diagram")
        plt.axis('off')
        plt.show()

    return img_rgb

# Extract regions from voronoi diagram based on colors - with visualization toggle
def extract_voronoi_regions(img, show_viz=False):
    # Define color ranges for buildings (purple) and green spaces
    # Adjusted to better match the uploaded image
    color_ranges = {
        'Buildings': ([80, 0, 80], [255, 150, 255]),  # Expanded purple range
        'GreenSpace': ([0, 80, 0], [180, 255, 180]),   # Expanded green range
        'Roads': ([0, 0, 0], [60, 60, 60])             # Expanded black range for roads/boundaries
    }

    masks = {}

    # Create the visualization only if requested
    if show_viz:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 4, 1)
        plt.imshow(img)
        plt.title("Original Voronoi Diagram")
        plt.axis('off')

    for idx, (region_type, (lower, upper)) in enumerate(color_ranges.items(), 2):
        lower = np.array(lower)
        upper = np.array(upper)
        mask = cv2.inRange(img, lower, upper)
        masks[region_type] = mask

        if show_viz and idx <= 4:  # Show plots for all region types if viz enabled
            plt.subplot(1, 4, idx)
            plt.imshow(mask, cmap='gray')
            plt.title(f"{region_type}")
            plt.axis('off')

    if show_viz:
        plt.tight_layout()
        plt.show()

    return masks

# Create polygons from masks - more efficient version
def create_polygons_from_masks(masks):
    region_polygons = {}

    for region_type, mask in masks.items():
        # Use contour detection to find region boundaries
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polygons = []
        for contour in contours:
            if cv2.contourArea(contour) > 50:  # Filter out small artifacts
                # Approximate the contour to simplify it
                epsilon = 0.01 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Convert to shapely polygon
                points = [(point[0][0], point[0][1]) for point in approx]
                if len(points) >= 3:  # Need at least 3 points for a valid polygon
                    try:
                        poly = Polygon(points)
                        if poly.is_valid and poly.area > 50:
                            polygons.append(poly)
                    except:
                        pass  # Skip invalid polygons

        region_polygons[region_type] = polygons

    return region_polygons

# Extract Voronoi edges from mask - more efficient
def extract_voronoi_edges(road_mask, max_edges=1000):
    # Thin the road mask to get centerlines
    kernel = np.ones((3,3), np.uint8)
    thinned_mask = cv2.erode(road_mask, kernel, iterations=1)

    # Find contours of the thinned roads
    contours, _ = cv2.findContours(thinned_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Convert contours to line segments with limit on edge count
    edges = []
    edge_count = 0

    for contour in contours:
        # Skip very small contours
        if len(contour) < 2:
            continue

        # Create line segments
        points = [tuple(point[0]) for point in contour]

        # If contour has many points, simplify it
        if len(points) > 10:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = [tuple(point[0]) for point in approx]

        # Create line segments from consecutive points
        for i in range(len(points)-1):
            edges.append((points[i], points[i+1]))
            edge_count += 1
            if edge_count >= max_edges:
                return edges

    return edges

# ===== VAE-INSPIRED BUILDING FOOTPRINT GENERATOR =====
class VAEInspiredGenerator:
    """VAE-inspired generator for building footprints - optimized version"""
    def __init__(self, latent_dim=8):
        self.latent_dim = latent_dim

    def generate_building_in_region(self, region, building_type='commercial'):
        """Generate a building footprint using VAE-like sampling within a region"""
        try:
            # Get region bounds
            minx, miny, maxx, maxy = region.bounds
            width = maxx - minx
            depth = maxy - miny

            # Skip if region is too small
            if width < 10 or depth < 10:
                return None

            # Calculate margins - simplified logic
            margin_factor = random.uniform(0.03, 0.15)
            margin = min(width, depth) * margin_factor
            building_width = width - 2 * margin
            building_depth = depth - 2 * margin

            # Skip if building would be too small
            if building_width < 5 or building_depth < 5:
                return None

            # Create base shape - always start with a rectangle
            base_rect = box(minx + margin, miny + margin, maxx - margin, maxy - margin)

            # Only add variations 50% of the time to reduce computation
            if random.random() < 0.5:
                # Simplify: Just two variation types instead of three
                variation_type = random.choice(['extension', 'subtraction'])

                if variation_type == 'extension':
                    # Add an extension to one side
                    side = random.choice(['left', 'right', 'top', 'bottom'])
                    ext_width = building_width * random.uniform(0.3, 0.6)
                    ext_depth = building_depth * random.uniform(0.3, 0.6)

                    if side == 'left':
                        ext_minx = max(minx, minx + margin - ext_width * 0.8)
                        ext_miny = miny + margin + random.uniform(0, building_depth - ext_depth)
                        extension = box(ext_minx, ext_miny, minx + margin, ext_miny + ext_depth)
                    elif side == 'right':
                        ext_minx = maxx - margin
                        ext_miny = miny + margin + random.uniform(0, building_depth - ext_depth)
                        extension = box(ext_minx, ext_miny, min(maxx, ext_minx + ext_width * 0.8), ext_miny + ext_depth)
                    elif side == 'top':
                        ext_minx = minx + margin + random.uniform(0, building_width - ext_width)
                        ext_miny = max(miny, miny + margin - ext_depth * 0.8)
                        extension = box(ext_minx, ext_miny, ext_minx + ext_width, miny + margin)
                    else:  # bottom
                        ext_minx = minx + margin + random.uniform(0, building_width - ext_width)
                        ext_miny = maxy - margin
                        extension = box(ext_minx, ext_miny, ext_minx + ext_width, min(maxy, ext_miny + ext_depth * 0.8))

                    poly = unary_union([base_rect, extension])

                else:  # subtraction - create courtyards
                    court_width = building_width * random.uniform(0.2, 0.4)
                    court_depth = building_depth * random.uniform(0.2, 0.4)
                    court_minx = minx + margin + (building_width - court_width) * random.uniform(0.3, 0.7)
                    court_miny = miny + margin + (building_depth - court_depth) * random.uniform(0.3, 0.7)

                    courtyard = box(court_minx, court_miny, court_minx + court_width, court_miny + court_depth)
                    poly = base_rect.difference(courtyard)
            else:
                # Just use the basic rectangle with slight randomness in size
                size_factor = random.uniform(0.9, 1.0)
                center_x = (minx + maxx) / 2
                center_y = (miny + maxy) / 2
                half_width = building_width * size_factor / 2
                half_depth = building_depth * size_factor / 2

                poly = box(center_x - half_width, center_y - half_depth,
                          center_x + half_width, center_y + half_depth)

            # Only rotate 20% of the time to reduce computation
            if random.random() < 0.2:
                angle = random.uniform(-15, 15)
                poly = rotate(poly, angle, origin='centroid')

                # Ensure building is within the region
                if not region.contains(poly):
                    # Revert to simpler shape if rotation caused issues
                    poly = box(minx + margin, miny + margin, maxx - margin, maxy - margin)

            return poly

        except Exception as e:
            # Fallback to simple rectangle without error message to reduce log output
            try:
                width = maxx - minx
                depth = maxy - miny
                margin = min(width, depth) * 0.1
                return box(minx + margin, miny + margin, maxx - margin, maxy - margin)
            except:
                return None

    # Modified for more efficient batch processing
    def generate_multiple_buildings_in_region(self, region, building_type, count=3, max_attempts=5):
        """Generate multiple buildings within a region - with attempt limit"""
        buildings = []
        if region.area < 100:  # Skip very small regions
            return buildings

        # For large regions, try to subdivide
        minx, miny, maxx, maxy = region.bounds
        width = maxx - minx
        depth = maxy - miny

        # Limit number of buildings based on region size
        if width > 100 and depth > 100 and count > 1:
            # Adjusted maximum count based on area to prevent excessive generation
            max_count = min(count, int(region.area / 2000) + 1)

            # Try to place multiple buildings with limit on attempts
            attempts = 0
            while len(buildings) < max_count and attempts < max_attempts:
                attempts += 1

                # Pick a subregion
                subregion_width = width * random.uniform(0.3, 0.6)
                subregion_depth = depth * random.uniform(0.3, 0.6)

                sub_minx = minx + random.uniform(0, width - subregion_width)
                sub_miny = miny + random.uniform(0, depth - subregion_depth)

                subregion = box(sub_minx, sub_miny,
                               sub_minx + subregion_width,
                               sub_miny + subregion_depth)

                if region.contains(subregion):
                    # Generate a building in this subregion
                    building = self.generate_building_in_region(subregion, building_type)
                    if building and building.is_valid and region.contains(building):
                        buildings.append(building)
        else:
            # Just generate one building
            building = self.generate_building_in_region(region, building_type)
            if building and building.is_valid:
                buildings.append(building)

        return buildings

# ===== GAN-INSPIRED BUILDING TYPE CLASSIFIER =====
# Optimized version with simplified logic
class GANInspiredClassifier:
    """GAN-inspired classifier for building types - optimized version"""

    def __init__(self):
        # Create simplified "learned" patterns for different building types
        self.patterns = {
            'commercial': {
                'height_mean': 40,
                'height_std': 20,
                'glass_prob': 0.7,
                'complex_prob': 0.6
            },
            'residential': {
                'height_mean': 25,
                'height_std': 10,
                'glass_prob': 0.3,
                'complex_prob': 0.4
            },
            'mixed': {  # Reduced to 3 types for simplicity
                'height_mean': 30,
                'height_std': 15,
                'glass_prob': 0.6,
                'complex_prob': 0.5
            }
        }

    def classify_region(self, region, nearby_regions=None):
        """Classify a region into a building type - simplified"""
        # Get region shape features - use fewer features
        area = region.area
        perimeter = region.length

        # Use simplified logic - just area-based classification
        if area > 3000:
            building_type = 'commercial'
        elif area < 1000:
            building_type = 'residential'
        else:
            building_type = 'mixed'

        # Add randomness for variety
        if random.random() < 0.3:  # 30% chance to choose randomly
            building_type = random.choice(list(self.patterns.keys()))

        return building_type, self.patterns[building_type]

# ===== DIFFUSION-INSPIRED HEIGHT GENERATOR =====
# Optimized with simpler logic
class DiffusionInspiredGenerator:
    """Diffusion-inspired generator for building heights and styles - optimized"""

    def __init__(self):
        # Simplified base patterns
        self.base_heights = {
            'commercial': (35, 15),  # (mean, std)
            'residential': (20, 8),
            'mixed': (25, 10)
        }

        # Simplified style patterns
        self.style_patterns = {
            'commercial': 0.7,  # Glass probability
            'residential': 0.3,
            'mixed': 0.5
        }

    def generate_building_properties(self, building_type, building_class_patterns):
        """Generate building height and style - simplified approach"""
        # Get height distribution parameters
        mean, std = self.base_heights[building_type]

        # Generate height with simplified noise
        height = np.random.normal(mean, std)
        height = max(10, height)  # Minimum height

        # Simplified style generation
        has_glass = random.random() < self.style_patterns[building_type]
        glass_sides = random.randint(1, 3) if has_glass else 0

        # Create style dict
        style = {
            'has_glass': has_glass,
            'glass_sides': glass_sides,
            'glass_color': 'blue' if has_glass else None,
            'modern_style': random.random() < 0.6,
            'complex_shape': random.random() < building_class_patterns['complex_prob']
        }

        return height, style

# ===== BUILDING GENERATION =====
def generate_buildings_with_genai(building_regions, batch_size=50):
    """Generate buildings in batches to reduce memory usage"""
    buildings = []
    start_time = time.time()

    # Initialize the GenAI components
    vae_generator = VAEInspiredGenerator(latent_dim=8)
    gan_classifier = GANInspiredClassifier()
    diffusion_generator = DiffusionInspiredGenerator()

    print("Initialized GenAI components for building generation")

    # Process regions in batches
    total_regions = len(building_regions)
    for batch_start in range(0, total_regions, batch_size):
        batch_end = min(batch_start + batch_size, total_regions)
        print(f"Processing building batch {batch_start+1}-{batch_end} of {total_regions}")

        batch_regions = building_regions[batch_start:batch_end]
        batch_buildings = []

        # Process each building region in the batch
        for i, region in enumerate(batch_regions):
            try:
                # Step 1: Use GAN to classify the building type
                building_type, type_patterns = gan_classifier.classify_region(region)

                # Step 2: Use VAE to generate building footprints
                # Limit building count based on region size
                building_count = max(1, min(3, int(region.area / 5000)))
                building_polys = vae_generator.generate_multiple_buildings_in_region(
                    region, building_type, count=building_count)

                for building_poly in building_polys:
                    if building_poly is None or not building_poly.is_valid:
                        continue

                    # Step 3: Use Diffusion model to generate height and style
                    height, style = diffusion_generator.generate_building_properties(
                        building_type, type_patterns)

                    # Create building data
                    building = {
                        'polygon': building_poly,
                        'height': height,
                        'type': building_type,
                        'style': style
                    }

                    batch_buildings.append(building)

            except Exception:
                # Skip error reporting to reduce output
                continue

        # Add batch buildings to main list
        buildings.extend(batch_buildings)

        # Force garbage collection between batches
        gc.collect()

    elapsed = time.time() - start_time
    print(f"Generated {len(buildings)} buildings in {elapsed:.2f} seconds")
    return buildings

# ===== GREEN SPACE GENERATOR =====
# Simplified to reduce computations
class GreenSpaceGenerator:
    """Generator for trees and vegetation in green spaces - optimized"""

    def __init__(self):
        self.tree_types = ['spherical', 'conical']  # Reduced variety
        self.tree_weights = [0.7, 0.3]  # Weights

    def generate_trees(self, green_regions, density_factor=1.0, batch_size=20, max_trees_per_region=30):
        """Generate trees within green regions - batched and limited"""
        trees = []
        start_time = time.time()

        # Process regions in batches
        total_regions = len(green_regions)
        for batch_start in range(0, total_regions, batch_size):
            batch_end = min(batch_start + batch_size, total_regions)
            print(f"Processing green space batch {batch_start+1}-{batch_end} of {total_regions}")

            batch_regions = green_regions[batch_start:batch_end]
            batch_trees = []

            for region in batch_regions:
                try:
                    area = region.area

                    # Determine number of trees based on area - with hard limit
                    num_trees = min(max(3, int(area / 1000 * density_factor)), max_trees_per_region)

                    # Get region bounds
                    minx, miny, maxx, maxy = region.bounds

                    # Simplified tree generation - fewer clusters
                    num_clusters = max(1, min(3, num_trees // 5))
                    cluster_centers = []

                    # Generate cluster centers
                    for _ in range(num_clusters):
                        # Maximum 3 attempts to place a cluster center
                        for attempt in range(3):
                            cx = random.uniform(minx, maxx)
                            cy = random.uniform(miny, maxy)
                            point = Point(cx, cy)

                            if region.contains(point):
                                cluster_centers.append((cx, cy))
                                break

                    # If no valid centers, use region centroid
                    if not cluster_centers:
                        centroid = region.centroid
                        cluster_centers.append((centroid.x, centroid.y))

                    # Generate trees around clusters
                    trees_per_cluster = max(1, num_trees // len(cluster_centers))

                    for center in cluster_centers:
                        cluster_radius = min(
                            np.sqrt(area / (np.pi * len(cluster_centers))),
                            min(maxx - minx, maxy - miny) / 3
                        )

                        for _ in range(trees_per_cluster):
                            # Randomize position around cluster center
                            angle = random.uniform(0, 2 * np.pi)
                            distance = random.uniform(0, cluster_radius)

                            x = center[0] + distance * np.cos(angle)
                            y = center[1] + distance * np.sin(angle)

                            point = Point(x, y)

                            if region.contains(point):
                                # Simplified tree properties
                                height = random.uniform(8, 15)
                                radius = random.uniform(3, 6)

                                tree_type = random.choices(
                                    self.tree_types,
                                    weights=self.tree_weights
                                )[0]

                                batch_trees.append({
                                    'x': x,
                                    'y': y,
                                    'height': height,
                                    'radius': radius,
                                    'type': tree_type
                                })

                except Exception:
                    # Skip error reporting to reduce output
                    continue

            # Add batch trees to main list
            trees.extend(batch_trees)

            # Force garbage collection between batches
            gc.collect()

        elapsed = time.time() - start_time
        print(f"Generated {len(trees)} trees in {elapsed:.2f} seconds")
        return trees

# ===== 3D VISUALIZATION FUNCTIONS =====
# Optimized to reduce memory usage

def create_building_3d_models(buildings, max_buildings=500):
    """Convert building data to 3D models - with limit on total buildings"""
    building_models = []

    # Limit the number of buildings to prevent memory issues
    if len(buildings) > max_buildings:
        print(f"Warning: Limiting visualization to {max_buildings} of {len(buildings)} buildings")
        # Take a representative sample
        buildings = random.sample(buildings, max_buildings)

    for building in buildings:
        polygon = building['polygon']
        height = building['height']
        building_type = building['type']
        style = building['style']

        # Extract coordinates from polygon - simplify if too complex
        coords = list(polygon.exterior.coords)
        if len(coords) > 20:  # Simplify complex polygons
            boundary = polygon.exterior
            length = boundary.length
            tolerance = length / 50  # Adaptive simplification
            simplified = boundary.simplify(tolerance, preserve_topology=True)
            coords = list(simplified.coords)

        # Generate vertices
        vertices = []

        # Bottom vertices (z=0)
        for x, y in coords[:-1]:  # skip the last point (same as first in shapely)
            vertices.append([x, y, 0])

        # Top vertices
        for x, y in coords[:-1]:
            vertices.append([x, y, height])

        # Determine which faces have glass - simplified
        glass_faces = []
        if style['has_glass']:
            num_sides = len(coords) - 1
            num_glass_sides = min(style['glass_sides'], num_sides)
            glass_faces = random.sample(range(num_sides), num_glass_sides)

        # Store the building model
        building_model = {
            'type': '3d_building',
            'vertices': np.array(vertices, dtype=FLOAT_PRECISION),  # Use lower precision
            'height': height,
            'building_type': building_type,
            'glass_faces': glass_faces,
            'glass_color': style.get('glass_color', 'blue')
        }

        building_models.append(building_model)

    return building_models

# Visualization function with Plotly - optimized for memory efficiency
# Fix for the visualize_3d_city function
def visualize_3d_city(building_models, trees, voronoi_edges, output_path=None,
                      max_trees=300, max_edges=200):
    """Create a 3D visualization with limits on object counts"""
    print("Creating 3D visualization...")
    # Create figure
    fig = go.Figure()

    # Sample trees if there are too many
    if len(trees) > max_trees:
        print(f"Sampling {max_trees} trees from {len(trees)} for visualization")
        trees = random.sample(trees, max_trees)

    # Sample edges if there are too many
    if len(voronoi_edges) > max_edges:
        print(f"Sampling {max_edges} edges from {len(voronoi_edges)} for visualization")
        voronoi_edges = random.sample(voronoi_edges, max_edges)

    # Add buildings with simplified geometry
    building_count = 0
    for building in building_models:
        vertices = building['vertices']
        glass_faces = building['glass_faces']
        glass_color = building['glass_color']

        # Convert glass color to RGBA
        glass_rgba = 'rgba(135, 206, 235, 0.7)'  # Default to light blue

        num_points = len(vertices) // 2

        # Skip buildings with too complex geometry
        if num_points > 20:
            continue

        building_count += 1
        if building_count > 300:  # Hard limit on buildings in visualization
            break

        # Create faces
        i_faces = []
        j_faces = []
        k_faces = []

        # Colors for different faces
        face_colors = []

        # Side faces - simplified approach
        for i in range(num_points):
            # First triangle
            i_faces.append(i)
            j_faces.append((i + 1) % num_points)
            k_faces.append(i + num_points)

            # Color (glass or white)
            if i in glass_faces:
                face_colors.append(glass_rgba)
            else:
                face_colors.append('white')

            # Second triangle
            i_faces.append((i + 1) % num_points)
            j_faces.append((i + 1) % num_points + num_points)
            k_faces.append(i + num_points)

            # Color (glass or white)
            if i in glass_faces:
                face_colors.append(glass_rgba)
            else:
                face_colors.append('white')

        # Add top face (always white) - simplified
        if num_points > 3:
            for i in range(2, num_points):
                i_faces.append(num_points)
                j_faces.append(num_points + i - 1)
                k_faces.append(num_points + i)
                face_colors.append('white')

        # Add building
        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=i_faces,
            j=j_faces,
            k=k_faces,
            facecolor=face_colors,
            flatshading=True,
            lighting=dict(ambient=0.8, diffuse=0.8),
            showlegend=False
        ))

    # Add trees with simplified geometry
    tree_count = 0
    for tree in trees:
        x, y, height, radius = tree['x'], tree['y'], tree['height'], tree['radius']
        tree_type = tree['type']

        tree_count += 1
        if tree_count > max_trees:  # Hard limit on trees rendered
            break

        # Use simplified tree geometry - only two types
        if tree_type == 'spherical':
            # Create simplified sphere (fewer points)
            u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:5j]  # Reduced resolution
            x_sphere = x + radius * np.cos(u) * np.sin(v)
            y_sphere = y + radius * np.sin(u) * np.sin(v)
            z_sphere = radius * np.cos(v) + 1  # Move slightly up from ground

            # Add tree as green sphere
            fig.add_trace(go.Surface(
                x=x_sphere,
                y=y_sphere,
                z=z_sphere,
                colorscale=[[0, '#228B22'], [1, '#32CD32']],  # Dark to light green
                showscale=False,
                showlegend=False
            ))

        else:  # conical - simplified
            # Create a simpler cone for coniferous trees
            theta = np.linspace(0, 2*np.pi, 8)  # Fewer points
            height_trunk = height * 0.2
            height_crown = height * 0.8

            # Add crown as a simplified cone
            # Trunk (brown cylinder)
            trunk_radius = radius * 0.15
            x_trunk = trunk_radius * np.cos(theta) + x
            y_trunk = trunk_radius * np.sin(theta) + y

            z_bottom = np.zeros_like(theta)
            z_top = np.ones_like(theta) * height_trunk

            # Add simplified trunk
            fig.add_trace(go.Mesh3d(
                x=np.concatenate([x_trunk, x_trunk]),
                y=np.concatenate([y_trunk, y_trunk]),
                z=np.concatenate([z_bottom, z_top]),
                i=[0, 0, 0, 0],  # Simplified indices
                j=[1, 2, 3, 4],
                k=[9, 10, 11, 12],
                color='#8B4513',  # SaddleBrown
                opacity=1.0,
                showlegend=False
            ))

            # Crown (green cone) - simplified
            crown_radius = radius
            cone_x = [x]  # Top point
            cone_y = [y]
            cone_z = [height_trunk + height_crown]

            # Add base circle - fewer points
            for t in theta:
                cone_x.append(x + crown_radius * np.cos(t))
                cone_y.append(y + crown_radius * np.sin(t))
                cone_z.append(height_trunk)

            # Create cone faces - simplified
            i_faces = []
            j_faces = []
            k_faces = []

            for i in range(1, len(cone_x) - 1):
                i_faces.append(0)  # Top vertex
                j_faces.append(i)
                k_faces.append(i + 1)

            # Close the cone
            i_faces.append(0)
            j_faces.append(len(cone_x) - 1)
            k_faces.append(1)

            fig.add_trace(go.Mesh3d(
                x=cone_x,
                y=cone_y,
                z=cone_z,
                i=i_faces,
                j=j_faces,
                k=k_faces,
                color='#006400',  # DarkGreen
                opacity=0.9,
                showlegend=False
            ))

    # Add Voronoi edges (overlay of original diagram) - with limit
    edge_count = 0
    for edge in voronoi_edges:
        start, end = edge

        edge_count += 1
        if edge_count > max_edges:  # Hard limit on edges rendered
            break

        fig.add_trace(go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[1, 1],  # Slightly above ground
            mode='lines',
            line=dict(color='black', width=3),
            showlegend=False
        ))

    # Add simplified ground plane
    if building_models:
        # Find bounds - use only first 100 buildings for efficiency
        all_vertices = np.vstack([b['vertices'] for b in building_models[:100]])
        x_min, x_max = all_vertices[:, 0].min(), all_vertices[:, 0].max()
        y_min, y_max = all_vertices[:, 1].min(), all_vertices[:, 1].max()

        # Add margin
        margin = max((x_max - x_min), (y_max - y_min)) * 0.2
        x_min -= margin
        x_max += margin
        y_min -= margin
        y_max += margin

        # Create ground plane
        fig.add_trace(go.Surface(
            x=np.array([[x_min, x_max], [x_min, x_max]]),
            y=np.array([[y_min, y_min], [y_max, y_max]]),
            z=np.zeros((2, 2)),
            colorscale=[[0, '#F5F5DC'], [1, '#FFFFF0']],  # Beige to light cream
            showscale=False,
            showlegend=False
        ))

    # Camera and scene setup - simplified
    fig.update_layout(
        title="Optimized 3D Urban Visualization",
        scene=dict(
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=0.8)
            ),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        ),
        margin=dict(r=10, l=10, b=10, t=30),
        scene_bgcolor='#F0F8FF',
        paper_bgcolor='#F0F8FF'
    )

    # Save if path provided - FIX HERE: Remove reference to undefined display_inline
    if output_path:
        pio.write_html(fig, file=output_path, auto_open=False)
        print(f"Visualization saved to {output_path}")

    # Return the figure for display in Colab
    return fig

# ===== SUBDIVIDE REGIONS FOR MORE BUILDINGS =====
def subdivide_building_regions(building_regions, max_regions=200):
    """Subdivide building regions to create more spaces for buildings - with limits"""
    print("Subdividing building regions...")
    start_time = time.time()

    # Limit input regions to prevent memory issues
    if len(building_regions) > max_regions:
        print(f"Limiting subdivision to {max_regions} regions")
        building_regions = random.sample(building_regions, max_regions)

    subdivided_regions = []

    for region in building_regions:
        if region.area < 500:  # Don't subdivide small regions
            subdivided_regions.append(region)
            continue

        # Simplified subdivision logic
        if region.area < 3000:
            parts = 2  # Fewer parts for medium regions
        else:
            parts = 3  # Fewer parts for large regions

        # Use simplified subdivision
        minx, miny, maxx, maxy = region.bounds
        width = maxx - minx
        depth = maxy - miny

        # Generate random points inside the region - fewer attempts
        points = []
        max_attempts = parts * 2  # Fewer attempts
        attempts = 0

        while len(points) < parts and attempts < max_attempts:
            attempts += 1
            x = minx + random.random() * width
            y = miny + random.random() * depth
            point = Point(x, y)
            if region.contains(point):
                points.append((x, y))

        if len(points) < 3:
            # Not enough points for Voronoi - keep original
            subdivided_regions.append(region)
            continue

        try:
            # Create Voronoi diagram
            vor = Voronoi(np.array(points))

            # Extract regions - simplified approach
            for region_idx in range(len(vor.point_region)):
                region_id = vor.point_region[region_idx]
                if region_id == -1:
                    continue

                # Get vertices of the Voronoi cell
                vertex_indices = vor.regions[region_id]
                if -1 in vertex_indices:  # Skip unbounded regions
                    continue

                # Create polygon
                vertices = [vor.vertices[i] for i in vertex_indices]
                if len(vertices) < 3:
                    continue

                try:
                    poly = Polygon(vertices)
                    if poly.is_valid and poly.area > 50:
                        # Intersect with original region
                        intersection = poly.intersection(region)
                        if intersection.area > 50:
                            if isinstance(intersection, Polygon):
                                subdivided_regions.append(intersection)
                            elif isinstance(intersection, MultiPolygon):
                                # Limit number of parts from a MultiPolygon
                                for p in list(intersection.geoms)[:2]:  # Take at most 2 parts
                                    if p.area > 50:
                                        subdivided_regions.append(p)
                except:
                    continue
        except:
            # Voronoi failed, just use the original region
            subdivided_regions.append(region)

    # If subdivision resulted in too few regions, add back some originals
    if len(subdivided_regions) < len(building_regions) * 0.7:
        # Add back original regions that weren't already added
        original_regions_to_add = [r for r in building_regions if not any(r.equals(sr) for sr in subdivided_regions)]
        subdivided_regions.extend(original_regions_to_add[:max_regions-len(subdivided_regions)])

    elapsed = time.time() - start_time
    print(f"Created {len(subdivided_regions)} subdivided regions in {elapsed:.2f} seconds")
    return subdivided_regions

# ===== MAIN FUNCTION =====
def generate_city_from_voronoi_with_genai(voronoi_image_path, output_path="genai_voronoi_3d_city.html",
                                         show_viz=False, max_buildings=300, max_trees=300):
    """Generate a 3D urban model from a Voronoi diagram - optimized version"""
    print("\n=== Initializing GenAI Voronoi Urban Generator (Optimized) ===")
    start_time = time.time()

    # Step 1: Load and process the Voronoi diagram
    img = load_and_process_image(voronoi_image_path, show_viz)

    # Step 2: Extract regions by color
    print("Extracting building and green space regions...")
    masks = extract_voronoi_regions(img, show_viz)

    # Step 3: Create polygons from masks
    print("Creating region polygons...")
    region_polygons = create_polygons_from_masks(masks)

    # Step 4: Extract Voronoi edges for overlay
    print("Extracting Voronoi edges...")
    voronoi_edges = extract_voronoi_edges(masks.get('Roads', np.zeros_like(masks['Buildings'])))

    # Step 5: Subdivide building regions for more density
    print("Subdividing building regions for higher density...")
    building_regions = region_polygons.get('Buildings', [])
    subdivided_building_regions = subdivide_building_regions(building_regions)

    # Step 6: Generate buildings using GenAI components
    print("Generating buildings with GenAI...")
    buildings = generate_buildings_with_genai(subdivided_building_regions)

    # Step 7: Create 3D building models with limits
    print("Creating 3D building models...")
    building_models = create_building_3d_models(buildings, max_buildings=max_buildings)

    # Step 8: Generate trees in green regions with the tree generator
    print("Generating trees...")
    green_regions = region_polygons.get('GreenSpace', [])
    green_space_generator = GreenSpaceGenerator()
    trees = green_space_generator.generate_trees(green_regions, density_factor=0.8)

    # Force garbage collection before visualization
    gc.collect()

    # Step 9: Visualize the results with limits
    print("Creating 3D visualization...")
    fig = visualize_3d_city(building_models, trees, voronoi_edges, output_path,
                          max_trees=max_trees, max_edges=200)

    total_time = time.time() - start_time
    print(f"\n=== Completed in {total_time:.2f} seconds ===")
    print(f"Generated {len(building_models)} buildings and {len(trees)} trees")
    print(f"Visualization saved to {output_path}")

    return building_models, trees, fig

if __name__ == "__main__":
    # Replace with your Voronoi diagram image path
    voronoi_image_path = "/content/drive/MyDrive/asd3.jpg"

    # Set visualization parameters for memory efficiency
    max_buildings = 300  # Limit number of buildings in visualization
    max_trees = 300      # Limit number of trees in visualization
    show_viz = False     # Set to True only if you want to see intermediate visualizations

    # Generate the 3D city with optimized parameters
    building_models, trees, fig = generate_city_from_voronoi_with_genai(
        voronoi_image_path,
        output_path="optimized_3d_city.html",
        show_viz=show_viz,
        max_buildings=max_buildings,
        max_trees=max_trees
    )

    # Display the figure directly in the Colab notebook
    fig.show()