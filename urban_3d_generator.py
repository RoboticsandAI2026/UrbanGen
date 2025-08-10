import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random
import os
import math
from skimage import measure
from scipy import ndimage
from shapely.geometry import Polygon, Point, LineString
import warnings
warnings.filterwarnings("ignore")

class Urban3DGenerator:
    def __init__(self, image_path=None):
        """
        Initialize the Urban 3D Generator.
        
        Args:
            image_path: Path to the voronoi diagram image (optional, can be set later)
        """
        self.image_path = image_path
        self.image = None
        
        # Define zone type colors (RGB) - Adjusted based on your image
        self.zone_colors = {
            'Mixed': (155, 89, 182),       # Purple (#9b59b6)
            'Industrial': (149, 165, 166), # Gray (#95a5a6)
            'Civic': (243, 156, 18),       # Orange (#f39c12)
            'Green': (46, 204, 113),       # Green (#2ecc71)
            'Commercial': (231, 76, 60),   # Red (#e74c3c) - Not in your image
            'Water': (52, 152, 219),       # Blue (#3498db)
            'Residential': (174, 247, 190) # Light green (#aef7be)
        }
        
        self.zones = []
        
        # Define building generation parameters for each zone type based on all requirements
        # FIXED: Increased probabilities and adjusted parameters to ensure elements appear
        self.zone_params = {
            'Commercial': {
                'height_range': (8, 20),
                'plot_coverage': (0.6, 0.8),
                'building_types': ['office_tower', 'retail_podium'],
                'max_buildings': 8,
                'building_probability': 1.0,
                'building_spacing': 10,
                'taller_buildings_in_center': True
            },
            'Mixed': {
                'height_range': (4, 10),
                'plot_coverage': (0.5, 0.7),
                'building_types': ['apartment', 'mixed_use'],
                'max_buildings': 6,
                'building_probability': 1.0,
                'add_courtyards': True,
                'varied_orientations': True,
                'commercial_ground_floor': True
            },
            'Green': {
                'height_range': (0, 2),
                'plot_coverage': (0.05, 0.2),
                'building_types': ['pavilion', 'gazebo'],
                'tree_density': 0.5,  # FIXED: Increased from 0 to 0.5
                'max_buildings': 2,
                'building_probability': 0.8,
                'add_walking_paths': True,
                'open_lawn_areas': True,
                'tree_size_variation': (0.8, 2.0)
            },
            'Water': {
                'height_range': (0, 0),
                'pier_probability': 0.8,
                'max_piers': 3,
                'add_waves': True,
                'add_boats': True,
                'shoreline_slope': 0.1
            },
            'Civic': {
                'height_range': (5, 12),
                'plot_coverage': (0.4, 0.6),
                'building_types': ['civic_center', 'museum', 'government'],
                'max_buildings': 3,
                'building_probability': 1.0,  # FIXED: Increased from 0.95 to 1.0
                'signature_buildings': True,
                'add_plazas': True,
                'formal_landscaping': True
            },
            'Residential': {
                'height_range': (2, 6),
                'plot_coverage': (0.4, 0.6),
                'building_types': ['house', 'apartment', 'townhouse'],
                'max_buildings': 12,
                'building_probability': 1.0,
                'regular_spacing': True,
                'add_driveways': True,
                'add_community_amenities': True,
                'green_space_ratio': 0.2
            },
            'Industrial': {
                'height_range': (1, 3),
                'plot_coverage': (0.7, 0.9),
                'building_types': ['warehouse', 'factory'],
                'max_buildings': 4,
                'building_probability': 1.0,  # FIXED: Increased from 0.95 to 1.0
                'wide_buildings': True,
                'add_loading_bays': True,
                'add_smokestacks': True,
                'add_storage_yards': True,
                'add_security_fencing': True,
                'landscaping_ratio': 0.15,
                'add_wide_roads': True
            }
        }
        
        # Define color palette for 3D rendering
        self.render_colors = {
            'Commercial': '#c0392b',
            'Mixed': '#8e44ad',
            'Green': '#27ae60',
            'Water': '#2980b9',
            'Civic': '#d35400',
            'Residential': '#aef7be',
            'Industrial': '#7f8c8d',
            'Tree': '#1e8449',
            'Road': '#34495e',
            'Pier': '#795548',
            'Plaza': '#bdc3c7',
            'Courtyard': '#f1c40f',
            'Lawn': '#2ecc71',
            'Driveway': '#7f8c8d',
            'Path': '#ecf0f1',
            'SecurityFence': '#95a5a6',
            'StorageYard': '#7f8c8d'
        }
        
        # Store generated elements
        self.buildings = []
        self.water_features = []
        self.trees = []
        self.paths = []
        self.plazas = []
        self.courtyards = []
        self.driveways = []
        self.storage_yards = []
        self.security_fences = []
        
        # Keep track of building footprints to prevent overlap
        self.building_footprints = []
        
        # If image path is provided, load it immediately
        if image_path:
            self.load_image(image_path)
    
    def load_image(self, image_path):
        """
        Load an image from the given path.
        
        Args:
            image_path: Path to the image file (supports PNG, JPG, WEBP, etc.)
        """
        print(f"Loading image from {image_path}...")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Try to load the image
        self.image = cv2.imread(image_path)
        
        if self.image is None:
            # Try with cv2.IMREAD_UNCHANGED for WEBP support
            self.image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            
        if self.image is None:
            raise ValueError(f"Could not read image at {image_path}. Unsupported format or corrupted file.")
        
        # Convert from BGR to RGB
        if len(self.image.shape) >= 3 and self.image.shape[2] == 3:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        elif len(self.image.shape) >= 3 and self.image.shape[2] == 4:
            # Handle RGBA images (like some WEBP)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGRA2RGB)
        
        self.image_path = image_path
        print(f"Image loaded: {self.image.shape}")
    
    def extract_zones(self):
        """Extract zones and their types from the image based on color."""
        if self.image is None:
            raise ValueError("No image loaded. Call load_image() first.")
        
        print("Extracting zones from image...")
        
        # Create a mask for each zone type
        zones = []
        img_height, img_width = self.image.shape[:2]
        
        # Define zone IDs dictionary to track IDs assigned to different types
        zone_id_counters = {
            'Mixed': 0,
            'Industrial': 0,
            'Civic': 0,
            'Green': 0,
            'Commercial': 0,
            'Water': 0,
            'Residential': 0
        }
        
        # Get zones based on labeled regions in the image
        for zone_type, color in self.zone_colors.items():
            # Create color mask with tolerance
            lower_bound = np.array([max(0, c - 30) for c in color])  # FIXED: Increased tolerance from 25 to 30
            upper_bound = np.array([min(255, c + 30) for c in color])
            mask = cv2.inRange(self.image, lower_bound, upper_bound)
            
            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Filter out very small contours
                if cv2.contourArea(contour) < 100:
                    continue
                    
                # Get bounding box and centroid
                x, y, w, h = cv2.boundingRect(contour)
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                else:
                    cx, cy = x + w // 2, y + h // 2
                
                # Create polygon from contour
                polygon = []
                for point in contour:
                    polygon.append((point[0][0], point[0][1]))
                
                # Try to extract zone ID from the image
                # Look for text near the center of the zone
                zone_id = None
                roi = self.image[max(0, cy-20):min(img_height, cy+20), 
                                max(0, cx-20):min(img_width, cx+20)]
                
                # In a real implementation, you would use OCR here
                # For now, we'll assign sequential IDs
                zone_id_counters[zone_type] += 1
                zone_id = zone_id_counters[zone_type]
                
                # Try to create shapely polygon (validate polygon)
                try:
                    shapely_polygon = Polygon(polygon)
                    if not shapely_polygon.is_valid:
                        # Try to fix invalid polygon
                        shapely_polygon = shapely_polygon.buffer(0)
                        if not shapely_polygon.is_valid:
                            print(f"Warning: Invalid polygon for {zone_type} zone. Skipping.")
                            continue
                except Exception as e:
                    print(f"Error creating polygon for {zone_type} zone: {e}")
                    continue
                
                # Create zone object
                zone = {
                    'type': zone_type,
                    'id': zone_id,
                    'contour': contour,
                    'polygon': polygon,
                    'bbox': (x, y, w, h),
                    'center': (cx, cy),
                    'area': cv2.contourArea(contour),
                    'shapely_polygon': shapely_polygon
                }
                zones.append(zone)
                
                # Also draw zone ID on image for debugging
                cv2.putText(self.image, f"#{zone_id}", (cx-15, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Sort zones by area (largest first)
        zones.sort(key=lambda z: z['area'], reverse=True)
        
        print(f"Found {len(zones)} zones")
        for zone_type in self.zone_colors.keys():
            count = sum(1 for z in zones if z['type'] == zone_type)
            print(f"  {zone_type}: {count} zones")
            
        self.zones = zones
        return zones
    
    def point_in_polygon(self, point, polygon):
        """Check if a point is inside a polygon."""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def buildings_overlap(self, building_corners):
        """
        Check if a proposed building overlaps with existing buildings.
        
        Args:
            building_corners: List of (x, y) tuples representing the corners of the building
        
        Returns:
            bool: True if there's an overlap, False otherwise
        """
        if not building_corners:
            return False
            
        # FIXED: Added buffer to make overlap detection less strict
        try:
            new_building = Polygon(building_corners).buffer(-1)  # 1-pixel buffer to allow closer placement
            
            # Check against all existing building footprints
            for footprint in self.building_footprints:
                try:
                    existing_building = Polygon(footprint)
                    if new_building.intersects(existing_building):
                        return True
                except Exception:
                    # Skip invalid polygons
                    continue
        except Exception:
            # If we can't create a valid polygon, assume no overlap
            return False
        
        return False
    
    def generate_buildings(self):
        """Generate 3D buildings for each zone based on zone type rules."""
        if not self.zones:
            print("No zones found. Running zone extraction first...")
            self.extract_zones()
        
        print("Generating 3D buildings for zones...")
        
        # Clear previous elements and footprints
        self.buildings = []
        self.water_features = []
        self.trees = []
        self.paths = []
        self.plazas = []
        self.courtyards = []
        self.driveways = []
        self.storage_yards = []
        self.security_fences = []
        self.building_footprints = []
        
        # Process zones in a specific order to handle larger zones first
        sorted_zones = sorted(self.zones, key=lambda z: z['area'], reverse=True)
        
        # FIXED: Process zone types in a specific order to ensure important features are placed first
        zone_type_order = ['Water', 'Green', 'Civic', 'Industrial', 'Residential', 'Mixed', 'Commercial']
        
        # First, process zones by type in the specified order
        for zone_type in zone_type_order:
            zones_of_type = [z for z in sorted_zones if z['type'] == zone_type]
            
            for zone in zones_of_type:
                # Skip if zone type not in parameters
                if zone_type not in self.zone_params:
                    continue
                    
                # Check if this zone will have buildings (based on probability)
                params = self.zone_params[zone_type]
                if random.random() > params.get('building_probability', 1.0):
                    print(f"Skipping buildings for {zone_type} zone (ID: {zone['id']}) based on probability")
                    continue
                    
                # Generate elements based on zone type
                if zone_type == 'Water':
                    self._generate_water_features(zone)
                elif zone_type == 'Green':
                    self._generate_green_features(zone)
                elif zone_type == 'Civic':
                    self._generate_civic_zone(zone)
                elif zone_type == 'Residential':
                    self._generate_residential_zone(zone)
                elif zone_type == 'Industrial':
                    self._generate_industrial_zone(zone)
                elif zone_type == 'Mixed':
                    self._generate_mixed_zone(zone)
                elif zone_type == 'Commercial':
                    self._generate_commercial_zone(zone)
        
        # Report generation stats
        print(f"Generated {len(self.buildings)} buildings")
        print(f"Generated {len(self.trees)} trees")
        print(f"Generated {len(self.water_features)} water features")
        print(f"Generated {len(self.plazas)} plazas/yards")
        print(f"Generated {len(self.paths)} paths")
    
    def _generate_commercial_zone(self, zone):
        """Generate commercial buildings with taller buildings in center."""
        polygon = zone['polygon']
        x, y, w, h = zone['bbox']
        center_x, center_y = zone['center']
        params = self.zone_params['Commercial']
        
        # Define buildable area (shapely polygon)
        zone_polygon = zone['shapely_polygon']
        
        # Determine number of buildings based on zone type and area
        area = zone['area']
        plot_coverage = random.uniform(params['plot_coverage'][0], params['plot_coverage'][1])
        total_building_area = area * plot_coverage
        
        max_buildings = min(params['max_buildings'], int(area / 1000) + 1)
        num_buildings = random.randint(max(1, max_buildings-2), max_buildings)
        
        # Calculate average building footprint
        avg_footprint = total_building_area / num_buildings
        
        # Maximum placement attempts
        max_attempts = 50
        placed_buildings = 0
        
        # Generate buildings
        for i in range(num_buildings):
            # Try to place building
            for attempt in range(max_attempts):
                # Determine building size
                width = random.uniform(avg_footprint**0.5 * 0.7, avg_footprint**0.5 * 1.3)
                depth = random.uniform(avg_footprint**0.5 * 0.7, avg_footprint**0.5 * 1.3)
                
                # Determine building position
                if i == 0:
                    # Place first commercial building near center
                    pos_x = center_x + random.uniform(-width/4, width/4)
                    pos_y = center_y + random.uniform(-depth/4, depth/4)
                else:
                    # Random position within zone
                    pos_x = x + random.uniform(width/2, w - width/2)
                    pos_y = y + random.uniform(depth/2, h - depth/2)
                
                # Random rotation (0, 90, 180, or 270 degrees for better placement)
                rotation = random.choice([0, 90, 180, 270])
                
                # Calculate building corners with rotation
                rotation_rad = np.radians(rotation)
                c, s = np.cos(rotation_rad), np.sin(rotation_rad)
                R = np.array([[c, -s], [s, c]])
                
                corners_base = np.array([
                    [-width/2, -depth/2],
                    [width/2, -depth/2],
                    [width/2, depth/2],
                    [-width/2, depth/2]
                ])
                
                # Apply rotation
                corners_rotated = np.dot(corners_base, R.T)
                
                # Add position offset
                building_corners = [(pos_x + x, pos_y + y) for x, y in corners_rotated]
                
                # Check if all corners are within zone
                try:
                    building_polygon = Polygon(building_corners)
                    # FIXED: Use contains_properly for stricter containment
                    all_corners_in_zone = zone_polygon.contains(building_polygon)
                except:
                    all_corners_in_zone = False
                
                # Check for building overlap
                overlap = self.buildings_overlap(building_corners)
                
                if all_corners_in_zone and not overlap:
                    # Determine building height based on position (taller in center)
                    min_height, max_height = params['height_range']
                    
                    if params.get('taller_buildings_in_center', False):
                        # Taller buildings toward center
                        dist_to_center = ((pos_x - center_x)**2 + (pos_y - center_y)**2)**0.5
                        max_dist = ((w/2)**2 + (h/2)**2)**0.5
                        height_factor = 1 - min(1, dist_to_center / max_dist) * 0.7
                        height = min_height + (max_height - min_height) * height_factor
                    else:
                        height = random.uniform(min_height, max_height)
                    
                    # Determine building type (office tower or retail podium)
                    if height > (min_height + max_height) / 2:
                        building_type = 'office_tower'
                    else:
                        building_type = 'retail_podium'
                    
                    # Create building
                    building = {
                        'type': building_type,
                        'zone_type': 'Commercial',
                        'zone_id': zone['id'],
                        'position': (pos_x, pos_y),
                        'size': (width, depth),
                        'height': height,
                        'rotation': rotation,
                        'corners': building_corners,
                        'stories': int(height)
                    }
                    
                    self.buildings.append(building)
                    self.building_footprints.append(building_corners)
                    placed_buildings += 1
                    break  # Successfully placed building
            
            # If we've tried too many times and can't place more buildings, break out
            if attempt == max_attempts - 1:
                break
        
        print(f"Placed {placed_buildings} commercial buildings in zone {zone['id']}")
    
    def _generate_mixed_zone(self, zone):
        """Generate mixed-use buildings with courtyards."""
        polygon = zone['polygon']
        x, y, w, h = zone['bbox']
        center_x, center_y = zone['center']
        params = self.zone_params['Mixed']
        
        # Define buildable area
        zone_polygon = zone['shapely_polygon']
        
        # Determine number of buildings
        area = zone['area']
        plot_coverage = random.uniform(params['plot_coverage'][0], params['plot_coverage'][1])
        total_building_area = area * plot_coverage
        
        # FIXED: Scale max_buildings based on area
        base_max_buildings = params['max_buildings']
        scaled_max_buildings = min(base_max_buildings, max(1, int(area / 1000)) + 1)
        num_buildings = random.randint(max(1, scaled_max_buildings-2), scaled_max_buildings)
        
        # Calculate average building footprint
        avg_footprint = total_building_area / num_buildings
        
        # Maximum placement attempts
        max_attempts = 50
        placed_buildings = 0
        
        # Generate buildings
        for i in range(num_buildings):
            for attempt in range(max_attempts):
                # Varied shapes for mixed-use buildings
                if params.get('varied_orientations', False) and random.random() < 0.3:
                    # L-shaped or irregular building
                    width = random.uniform(avg_footprint**0.5 * 0.8, avg_footprint**0.5 * 1.2)
                    depth = random.uniform(avg_footprint**0.5 * 0.8, avg_footprint**0.5 * 1.2)
                    
                    # Create L-shape by removing one corner
                    corner_to_remove = random.randint(0, 3)
                    corner_inset = min(width, depth) * 0.4
                    
                    if corner_to_remove == 0:
                        corners_base = np.array([
                            [-width/2 + corner_inset, -depth/2 + corner_inset],
                            [width/2, -depth/2],
                            [width/2, depth/2],
                            [-width/2, depth/2],
                            [-width/2, -depth/2 + corner_inset]
                        ])
                    elif corner_to_remove == 1:
                        corners_base = np.array([
                            [-width/2, -depth/2],
                            [width/2 - corner_inset, -depth/2 + corner_inset],
                            [width/2, depth/2],
                            [-width/2, depth/2],
                            [-width/2, -depth/2]
                        ])
                    elif corner_to_remove == 2:
                        corners_base = np.array([
                            [-width/2, -depth/2],
                            [width/2, -depth/2],
                            [width/2 - corner_inset, depth/2 - corner_inset],
                            [-width/2, depth/2],
                            [-width/2, -depth/2]
                        ])
                    else:
                        corners_base = np.array([
                            [-width/2, -depth/2],
                            [width/2, -depth/2],
                            [width/2, depth/2],
                            [-width/2 + corner_inset, depth/2 - corner_inset],
                            [-width/2, -depth/2]
                        ])
                else:
                    # Standard rectangular building
                    # FIXED: Scale building size better for mixed-use zones
                    width = random.uniform(avg_footprint**0.5 * 0.7, avg_footprint**0.5 * 1.3)
                    depth = random.uniform(avg_footprint**0.5 * 0.7, avg_footprint**0.5 * 1.3)
                    
                    corners_base = np.array([
                        [-width/2, -depth/2],
                        [width/2, -depth/2],
                        [width/2, depth/2],
                        [-width/2, depth/2]
                    ])
                
                # Random position (ensure we're inside the zone bounds)
                min_x = x + width/2
                max_x = x + w - width/2
                min_y = y + depth/2
                max_y = y + h - depth/2
                
                # Make sure we have valid bounds
                if min_x >= max_x:
                    min_x, max_x = x, x + w
                if min_y >= max_y:
                    min_y, max_y = y, y + h
                
                pos_x = random.uniform(min_x, max_x)
                pos_y = random.uniform(min_y, max_y)
                
                # Random rotation for varied orientations
                if params.get('varied_orientations', False):
                    rotation = random.uniform(0, 360)
                else:
                    rotation = random.choice([0, 90, 180, 270])
                
                # Apply rotation
                rotation_rad = np.radians(rotation)
                c, s = np.cos(rotation_rad), np.sin(rotation_rad)
                R = np.array([[c, -s], [s, c]])
                corners_rotated = np.dot(corners_base, R.T)
                
                # Add position offset
                building_corners = [(pos_x + dx, pos_y + dy) for dx, dy in corners_rotated]
                
                # Check if all corners are within zone
                try:
                    building_polygon = Polygon(building_corners)
                    all_corners_in_zone = zone_polygon.contains(building_polygon)
                except:
                    all_corners_in_zone = False
                
                # Check for building overlap
                overlap = self.buildings_overlap(building_corners)
                
                if all_corners_in_zone and not overlap:
                    # Determine building height
                    min_height, max_height = params['height_range']
                    height = random.uniform(min_height, max_height)
                    
                    # Create building
                    building = {
                        'type': random.choice(params['building_types']),
                        'zone_type': 'Mixed',
                        'zone_id': zone['id'],
                        'position': (pos_x, pos_y),
                        'size': (width, depth),
                        'height': height,
                        'rotation': rotation,
                        'corners': building_corners,
                        'stories': int(height),
                        'commercial_ground_floor': params.get('commercial_ground_floor', False)
                    }
                    
                    self.buildings.append(building)
                    self.building_footprints.append(building_corners)
                    placed_buildings += 1
                    
                    # Add a courtyard if enabled and building is large enough
                    if params.get('add_courtyards', False) and random.random() < 0.4 and width > 15 and depth > 15:
                        courtyard_width = width * 0.3
                        courtyard_depth = depth * 0.3
                        
                        courtyard = {
                            'position': (pos_x, pos_y),
                            'size': (courtyard_width, courtyard_depth),
                            'rotation': rotation,
                            'zone_id': zone['id']
                        }
                        
                        self.courtyards.append(courtyard)
                    
                    break  # Successfully placed building
            
            # If we've tried too many times and can't place more buildings, break out
            if attempt == max_attempts - 1:
                break
        
        print(f"Placed {placed_buildings} mixed-use buildings in zone {zone['id']}")
    
    def _generate_green_features(self, zone):
        """Generate trees, paths, and small structures for green zones."""
        polygon = zone['polygon']
        x, y, w, h = zone['bbox']
        params = self.zone_params['Green']
        area = zone['area']
        zone_polygon = zone['shapely_polygon']
        
        # FIXED: Generate trees with higher density and ensure they appear
        if params['tree_density'] > 0:
            # Calculate number of trees based on density and area
            tree_count = int(area * params['tree_density'] / 400)  # Adjusted to create more trees
            tree_count = min(max(5, tree_count), 100)  # Limit number of trees
            
            placed_trees = 0
            for _ in range(tree_count):
                for attempt in range(10):
                    # Random position within zone
                    pos_x = x + random.uniform(0, w)
                    pos_y = y + random.uniform(0, h)
                    
                    # Check if position is in zone
                    try:
                        point = Point(pos_x, pos_y)
                        if zone_polygon.contains(point):
                            # Vary tree size based on parameters
                            size_factor = random.uniform(
                                params.get('tree_size_variation', (0.8, 2.0))[0],
                                params.get('tree_size_variation', (0.8, 2.0))[1]
                            )
                            
                            tree_height = 2 * size_factor
                            trunk_height = tree_height * 0.4
                            crown_radius = tree_height * 0.3
                            
                            tree = {
                                'position': (pos_x, pos_y),
                                'trunk_height': trunk_height,
                                'crown_radius': crown_radius,
                                'tree_height': tree_height,
                                'zone_id': zone['id']
                            }
                            
                            self.trees.append(tree)
                            placed_trees += 1
                            break
                    except:
                        continue
            
            print(f"Placed {placed_trees} trees in green zone {zone['id']}")
        
        # Generate open lawn areas if enabled
        if params.get('open_lawn_areas', False):
            # Create 1-3 lawn areas
            num_lawns = random.randint(1, 3)
            
            for _ in range(num_lawns):
                # Random position and size for lawn
                lawn_width = random.uniform(w * 0.2, w * 0.4)
                lawn_depth = random.uniform(h * 0.2, h * 0.4)
                
                pos_x = x + random.uniform(lawn_width/2, w - lawn_width/2)
                pos_y = y + random.uniform(lawn_depth/2, h - lawn_depth/2)
                
                # Check if lawn is in zone (simplified)
                try:
                    lawn_polygon = Polygon([
                        (pos_x - lawn_width/2, pos_y - lawn_depth/2),
                        (pos_x + lawn_width/2, pos_y - lawn_depth/2),
                        (pos_x + lawn_width/2, pos_y + lawn_depth/2),
                        (pos_x - lawn_width/2, pos_y + lawn_depth/2)
                    ])
                    
                    if zone_polygon.contains(lawn_polygon):
                        lawn = {
                            'type': 'lawn',
                            'position': (pos_x, pos_y),
                            'size': (lawn_width, lawn_depth),
                            'zone_id': zone['id']
                        }
                        
                        # Add to plazas list (reusing the data structure)
                        self.plazas.append(lawn)
                except:
                    continue
        
        # Generate walking paths if enabled
        if params.get('add_walking_paths', False):
            # Create 1-3 paths through the zone
            num_paths = random.randint(1, 3)
            
            for _ in range(num_paths):
                # Generate path points
                num_points = random.randint(3, 6)
                path_points = []
                
                # Start with points on the zone boundary
                for i in range(num_points):
                    if i == 0 or i == num_points - 1:
                        # Get a point on the boundary
                        boundary_points = np.array(polygon)
                        rand_idx = random.randint(0, len(boundary_points) - 1)
                        point = (boundary_points[rand_idx][0], boundary_points[rand_idx][1])
                    else:
                        # Random point within the zone
                        pos_x = x + random.uniform(0, w)
                        pos_y = y + random.uniform(0, h)
                        point = (pos_x, pos_y)
                    
                    path_points.append(point)
                
                # Create a curved path
                path = {
                    'type': 'path',
                    'points': path_points,
                    'width': random.uniform(1, 3),
                    'zone_id': zone['id']
                }
                
                self.paths.append(path)
        
        # Generate small structures if enabled
        if params['max_buildings'] > 0:
            # Determine number of structures
            structure_count = random.randint(0, params['max_buildings'])
            
            for _ in range(structure_count):
                for attempt in range(20):
                    # Small structures for recreation
                    width = random.uniform(3, 8)
                    depth = random.uniform(3, 8)
                    
                    # Random position
                    pos_x = x + random.uniform(width, w - width)
                    pos_y = y + random.uniform(depth, h - depth)
                    
                    # Calculate corners
                    corners_base = np.array([
                        [-width/2, -depth/2],
                        [width/2, -depth/2],
                        [width/2, depth/2],
                        [-width/2, depth/2]
                    ])
                    
                    # Random rotation
                    rotation = random.uniform(0, 360)
                    rotation_rad = np.radians(rotation)
                    c, s = np.cos(rotation_rad), np.sin(rotation_rad)
                    R = np.array([[c, -s], [s, c]])
                    corners_rotated = np.dot(corners_base, R.T)
                    
                    # Add position offset
                    building_corners = [(pos_x + dx, pos_y + dy) for dx, dy in corners_rotated]
                    
                    # Check if all corners are in zone and not overlapping
                    try:
                        building_polygon = Polygon(building_corners)
                        all_corners_in_zone = zone_polygon.contains(building_polygon)
                    except:
                        all_corners_in_zone = False
                    
                    overlap = self.buildings_overlap(building_corners)
                    
                    if all_corners_in_zone and not overlap:
                        # Low height for green zone structures
                        min_height, max_height = params['height_range']
                        height = random.uniform(min_height, max_height)
                        
                        # Create pavilion or gazebo
                        building = {
                            'type': random.choice(params['building_types']),
                            'zone_type': 'Green',
                            'zone_id': zone['id'],
                            'position': (pos_x, pos_y),
                            'size': (width, depth),
                            'height': height,
                            'rotation': rotation,
                            'corners': building_corners,
                            'stories': 1
                        }
                        
                        self.buildings.append(building)
                        self.building_footprints.append(building_corners)
                        break
    
    def _generate_water_features(self, zone):
        """Generate water surface with piers, boats, and waves."""
        polygon = zone['polygon']
        x, y, w, h = zone['bbox']
        params = self.zone_params['Water']
        
        # Create water surface
        water = {
            'type': 'water_surface',
            'polygon': polygon,
            'position': (x + w/2, y + h/2),
            'size': (w, h),
            'add_waves': params.get('add_waves', False),
            'zone_id': zone['id']
        }
        self.water_features.append(water)
        
        # Add shoreline with gentle slope if enabled
        if params.get('shoreline_slope', 0) > 0:
            # Create a simplified shoreline by shrinking the polygon slightly
            shoreline = {
                'type': 'shoreline',
                'polygon': polygon,
                'slope': params.get('shoreline_slope', 0.1),
                'zone_id': zone['id']
            }
            self.water_features.append(shoreline)
        
        # Generate piers if enabled
        if params.get('max_piers', 0) > 0 and random.random() < params.get('pier_probability', 0):
            pier_count = random.randint(1, params['max_piers'])
            
            for _ in range(pier_count):
                # Find a point on the polygon perimeter
                perimeter_index = random.randint(0, len(polygon) - 1)
                start_x, start_y = polygon[perimeter_index]
                
                # Find nearest point outside polygon to determine direction
                outside_x = start_x + random.uniform(-10, 10)
                outside_y = start_y + random.uniform(-10, 10)
                
                # Make sure the point is outside
                for _ in range(10):
                    if not self.point_in_polygon((outside_x, outside_y), polygon):
                        break
                    outside_x = start_x + random.uniform(-15, 15)
                    outside_y = start_y + random.uniform(-15, 15)
                
                # Calculate direction into the water
                dir_x = start_x - outside_x
                dir_y = start_y - outside_y
                dist = (dir_x**2 + dir_y**2)**0.5
                if dist > 0:
                    dir_x /= dist
                    dir_y /= dist
                
                # Create pier
                pier_length = random.uniform(5, 15)
                pier_width = random.uniform(2, 4)
                
                end_x = start_x + dir_x * pier_length
                end_y = start_y + dir_y * pier_length
                
                # Check if end point is in water
                if self.point_in_polygon((end_x, end_y), polygon):
                    pier = {
                        'type': 'pier',
                        'start': (start_x, start_y),
                        'end': (end_x, end_y),
                        'width': pier_width,
                        'height': 0.5,  # Just above water level
                        'zone_id': zone['id']
                    }
                    self.water_features.append(pier)
                    
                    # Add a boat at the end of the pier if enabled
                    if params.get('add_boats', False) and random.random() < 0.7:
                        boat = {
                            'type': 'boat',
                            'position': (end_x + dir_x * 2, end_y + dir_y * 2),
                            'size': (random.uniform(2, 4), random.uniform(3, 7)),
                            'rotation': math.atan2(dir_y, dir_x) + random.uniform(-0.5, 0.5),
                            'zone_id': zone['id']
                        }
                        self.water_features.append(boat)
        
        # Add random boats if enabled
        if params.get('add_boats', False):
            num_boats = random.randint(1, 3)
            zone_polygon = Polygon(polygon)
            
            for _ in range(num_boats):
                # Random position inside water zone
                for attempt in range(10):
                    pos_x = x + random.uniform(w * 0.2, w * 0.8)
                    pos_y = y + random.uniform(h * 0.2, h * 0.8)
                    
                    if zone_polygon.contains(Point(pos_x, pos_y)):
                        boat = {
                            'type': 'boat',
                            'position': (pos_x, pos_y),
                            'size': (random.uniform(2, 4), random.uniform(3, 7)),
                            'rotation': random.uniform(0, 2 * math.pi),
                            'zone_id': zone['id']
                        }
                        self.water_features.append(boat)
                        break
    
    def _generate_civic_zone(self, zone):
        """Generate civic buildings with plazas and formal landscaping."""
        polygon = zone['polygon']
        x, y, w, h = zone['bbox']
        center_x, center_y = zone['center']
        params = self.zone_params['Civic']
        zone_polygon = zone['shapely_polygon']
        
        # FIXED: Better adjust for small civic zones
        area = zone['area']
        plot_coverage = random.uniform(params['plot_coverage'][0], params['plot_coverage'][1])
        total_building_area = area * plot_coverage
        
        max_buildings = min(params['max_buildings'], max(1, int(area / 2000) + 1))
        num_buildings = random.randint(1, max_buildings)
        
        placed_buildings = 0
        
        # Generate signature building first if enabled
        if params.get('signature_buildings', False):
            # Larger, more distinctive building in a prominent position
            signature_width = min(w * random.uniform(0.3, 0.5), 30)  # Cap max width
            signature_depth = min(h * random.uniform(0.3, 0.5), 30)  # Cap max depth
            
            # Try to place near center
            for attempt in range(20):
                # Position near center with some variation
                pos_x = center_x + random.uniform(-w * 0.15, w * 0.15)
                pos_y = center_y + random.uniform(-h * 0.15, h * 0.15)
                
                # Calculate corners
                corners_base = np.array([
                    [-signature_width/2, -signature_depth/2],
                    [signature_width/2, -signature_depth/2],
                    [signature_width/2, signature_depth/2],
                    [-signature_width/2, signature_depth/2]
                ])
                
                # Formal orientation (90-degree increments)
                rotation = random.choice([0, 90, 180, 270])
                rotation_rad = np.radians(rotation)
                c, s = np.cos(rotation_rad), np.sin(rotation_rad)
                R = np.array([[c, -s], [s, c]])
                corners_rotated = np.dot(corners_base, R.T)
                
                # Add position offset
                building_corners = [(pos_x + dx, pos_y + dy) for dx, dy in corners_rotated]
                
                # Check if all corners are in zone
                try:
                    building_polygon = Polygon(building_corners)
                    all_corners_in_zone = zone_polygon.contains(building_polygon)
                except:
                    all_corners_in_zone = False
                
                if all_corners_in_zone:
                    # Taller height for signature building
                    min_height, max_height = params['height_range']
                    height = max_height * random.uniform(0.8, 1.0)
                    
                    # Create signature building
                    building = {
                        'type': 'civic_center',
                        'zone_type': 'Civic',
                        'zone_id': zone['id'],
                        'position': (pos_x, pos_y),
                        'size': (signature_width, signature_depth),
                        'height': height,
                        'rotation': rotation,
                        'corners': building_corners,
                        'stories': int(height),
                        'is_signature': True
                    }
                    
                    self.buildings.append(building)
                    self.building_footprints.append(building_corners)
                    placed_buildings += 1
                    
                    # Add plaza in front of signature building if enabled
                    if params.get('add_plazas', False):
                        # Calculate plaza position in front of building
                        plaza_dir = random.choice([0, 1, 2, 3])  # Direction from building
                        
                        if plaza_dir == 0:  # North
                            plaza_x = pos_x
                            plaza_y = pos_y + signature_depth * 0.7
                        elif plaza_dir == 1:  # East
                            plaza_x = pos_x + signature_width * 0.7
                            plaza_y = pos_y
                        elif plaza_dir == 2:  # South
                            plaza_x = pos_x
                            plaza_y = pos_y - signature_depth * 0.7
                        else:  # West
                            plaza_x = pos_x - signature_width * 0.7
                            plaza_y = pos_y
                        
                        plaza_width = signature_width * 0.8
                        plaza_depth = signature_depth * 0.6
                        
                        plaza = {
                            'type': 'plaza',
                            'position': (plaza_x, plaza_y),
                            'size': (plaza_width, plaza_depth),
                            'rotation': rotation,
                            'zone_id': zone['id']
                        }
                        
                        self.plazas.append(plaza)
                    
                    # Reduce remaining buildings to allocate space
                    num_buildings = max(0, num_buildings - 1)
                    break
        
        # Calculate average building footprint
        avg_footprint = total_building_area / (num_buildings + 1)  # +1 to account for signature building
        
        # Generate additional buildings
        for i in range(num_buildings):
            for attempt in range(20):
                # FIXED: Scale building size better based on zone size
                width = min(random.uniform(avg_footprint**0.5 * 0.7, avg_footprint**0.5 * 1.2), w * 0.5)
                depth = min(random.uniform(avg_footprint**0.5 * 0.7, avg_footprint**0.5 * 1.2), h * 0.5)
                
                # Random position
                pos_x = x + random.uniform(width/2, w - width/2)
                pos_y = y + random.uniform(depth/2, h - depth/2)
                
                # Formal orientation (90-degree increments)
                rotation = random.choice([0, 90, 180, 270])
                
                # Calculate corners
                rotation_rad = np.radians(rotation)
                c, s = np.cos(rotation_rad), np.sin(rotation_rad)
                R = np.array([[c, -s], [s, c]])
                
                corners_base = np.array([
                    [-width/2, -depth/2],
                    [width/2, -depth/2],
                    [width/2, depth/2],
                    [-width/2, depth/2]
                ])
                
                corners_rotated = np.dot(corners_base, R.T)
                building_corners = [(pos_x + dx, pos_y + dy) for dx, dy in corners_rotated]
                
                # Check if all corners are in zone and not overlapping
                try:
                    building_polygon = Polygon(building_corners)
                    all_corners_in_zone = zone_polygon.contains(building_polygon)
                except:
                    all_corners_in_zone = False
                
                overlap = self.buildings_overlap(building_corners)
                
                if all_corners_in_zone and not overlap:
                    # Determine building height
                    min_height, max_height = params['height_range']
                    height = random.uniform(min_height, max_height * 0.8)  # Slightly lower than signature
                    
                    # Create building
                    building = {
                        'type': random.choice(['museum', 'government']),
                        'zone_type': 'Civic',
                        'zone_id': zone['id'],
                        'position': (pos_x, pos_y),
                        'size': (width, depth),
                        'height': height,
                        'rotation': rotation,
                        'corners': building_corners,
                        'stories': int(height)
                    }
                    
                    self.buildings.append(building)
                    self.building_footprints.append(building_corners)
                    placed_buildings += 1
                    break
        
        print(f"Placed {placed_buildings} civic buildings in zone {zone['id']}")
        
        # Add formal landscaping if enabled
        if params.get('formal_landscaping', False):
            # Add some trees in formal arrangements
            tree_count = int(area * 0.0002)
            tree_count = min(max(4, tree_count), 30)
            
            # Create rows or clusters of trees
            arrangement = random.choice(['row', 'cluster', 'perimeter'])
            
            if arrangement == 'row':
                # Create rows of trees along axes
                for i in range(tree_count):
                    axis = i % 2  # Alternate between x and y axes
                    position = i // 2
                    
                    if axis == 0:
                        # Row along x-axis
                        pos_x = x + w * (0.2 + (position % 3) * 0.3)
                        pos_y = y + h * (0.2 + (position // 3) * 0.3)
                    else:
                        # Row along y-axis
                        pos_x = x + w * (0.2 + (position // 3) * 0.3)
                        pos_y = y + h * (0.2 + (position % 3) * 0.3)
                    
                    # Check if position is in zone
                    if zone_polygon.contains(Point(pos_x, pos_y)):
                        tree = {
                            'position': (pos_x, pos_y),
                            'trunk_height': 1.5,
                            'crown_radius': 1.0,
                            'tree_height': 3.0,
                            'formal': True,
                            'zone_id': zone['id']
                        }
                        
                        self.trees.append(tree)
            
            elif arrangement == 'cluster':
                # Create clusters of trees
                num_clusters = random.randint(2, 4)
                trees_per_cluster = tree_count // num_clusters
                
                for c in range(num_clusters):
                    # Cluster center
                    cluster_x = x + random.uniform(w * 0.2, w * 0.8)
                    cluster_y = y + random.uniform(h * 0.2, h * 0.8)
                    
                    for t in range(trees_per_cluster):
                        # Position within cluster
                        angle = 2 * math.pi * t / trees_per_cluster
                        radius = random.uniform(3, 7)
                        
                        pos_x = cluster_x + radius * math.cos(angle)
                        pos_y = cluster_y + radius * math.sin(angle)
                        
                        # Check if position is in zone
                        if zone_polygon.contains(Point(pos_x, pos_y)):
                            tree = {
                                'position': (pos_x, pos_y),
                                'trunk_height': 1.5,
                                'crown_radius': 1.0,
                                'tree_height': 3.0,
                                'formal': True,
                                'zone_id': zone['id']
                            }
                            
                            self.trees.append(tree)
            
            else:  # perimeter
                # Trees along the perimeter
                perimeter_points = len(polygon)
                stride = max(1, perimeter_points // tree_count)
                
                for i in range(0, perimeter_points, stride):
                    pos_x, pos_y = polygon[i]
                    
                    # Move slightly inward from the boundary
                    vector_to_center = (center_x - pos_x, center_y - pos_y)
                    dist = (vector_to_center[0]**2 + vector_to_center[1]**2)**0.5
                    
                    if dist > 0:
                        unit_vector = (vector_to_center[0] / dist, vector_to_center[1] / dist)
                        pos_x += unit_vector[0] * 3
                        pos_y += unit_vector[1] * 3
                    
                    # Check if position is in zone
                    if zone_polygon.contains(Point(pos_x, pos_y)):
                        tree = {
                            'position': (pos_x, pos_y),
                            'trunk_height': 1.5,
                            'crown_radius': 1.0,
                            'tree_height': 3.0,
                            'formal': True,
                            'zone_id': zone['id']
                        }
                        
                        self.trees.append(tree)
    
    def _generate_residential_zone(self, zone):
        """Generate residential buildings with varied housing types."""
        polygon = zone['polygon']
        x, y, w, h = zone['bbox']
        params = self.zone_params['Residential']
        zone_polygon = zone['shapely_polygon']
        
        # Determine building distribution
        area = zone['area']
        plot_coverage = random.uniform(params['plot_coverage'][0], params['plot_coverage'][1])
        total_building_area = area * plot_coverage
        
        # Set aside area for green space if specified
        buildable_area = area * (1 - params.get('green_space_ratio', 0))
        
        # Determine mix of housing types
        # FIXED: Scale max_buildings based on area
        max_buildings = min(params['max_buildings'], max(2, int(area / 500) + 1))
        num_buildings = random.randint(max(2, max_buildings-3), max_buildings)
        
        # Determine how many of each type
        num_houses = int(num_buildings * 0.6)  # 60% single-family homes
        num_apartments = num_buildings - num_houses  # 40% apartment buildings
        
        # Calculate average footprints
        avg_house_footprint = 80  # Smaller footprint for houses
        avg_apt_footprint = (buildable_area * plot_coverage - num_houses * avg_house_footprint) / max(1, num_apartments)
        
        placed_buildings = 0
        
        # Generate apartment buildings first (larger structures)
        for i in range(num_apartments):
            for attempt in range(20):
                # Apartment buildings are larger
                width = random.uniform(avg_apt_footprint**0.5 * 0.8, avg_apt_footprint**0.5 * 1.2)
                depth = random.uniform(avg_apt_footprint**0.5 * 0.8, avg_apt_footprint**0.5 * 1.2)
                
                # Random position with regular spacing if enabled
                if params.get('regular_spacing', False):
                    # Create a grid-like arrangement
                    grid_size = int(num_buildings**0.5) + 1
                    row = i % grid_size
                    col = i // grid_size
                    
                    # Position with some jitter
                    pos_x = x + w * (0.1 + 0.8 * row / grid_size) + random.uniform(-w * 0.05, w * 0.05)
                    pos_y = y + h * (0.1 + 0.8 * col / grid_size) + random.uniform(-h * 0.05, h * 0.05)
                else:
                    # Random position
                    pos_x = x + random.uniform(width/2, w - width/2)
                    pos_y = y + random.uniform(depth/2, h - depth/2)
                
                # Calculate corners
                rotation = random.choice([0, 90, 180, 270])
                rotation_rad = np.radians(rotation)
                c, s = np.cos(rotation_rad), np.sin(rotation_rad)
                R = np.array([[c, -s], [s, c]])
                
                corners_base = np.array([
                    [-width/2, -depth/2],
                    [width/2, -depth/2],
                    [width/2, depth/2],
                    [-width/2, depth/2]
                ])
                
                corners_rotated = np.dot(corners_base, R.T)
                building_corners = [(pos_x + dx, pos_y + dy) for dx, dy in corners_rotated]
                
                # Check if all corners are in zone and not overlapping
                try:
                    building_polygon = Polygon(building_corners)
                    all_corners_in_zone = zone_polygon.contains(building_polygon)
                except:
                    all_corners_in_zone = False
                
                overlap = self.buildings_overlap(building_corners)
                
                if all_corners_in_zone and not overlap:
                    # Apartment buildings are taller
                    min_height, max_height = params['height_range']
                    height = random.uniform(min_height + 1, max_height)
                    
                    # Create apartment building
                    building = {
                        'type': 'apartment',
                        'zone_type': 'Residential',
                        'zone_id': zone['id'],
                        'position': (pos_x, pos_y),
                        'size': (width, depth),
                        'height': height,
                        'rotation': rotation,
                        'corners': building_corners,
                        'stories': int(height)
                    }
                    
                    self.buildings.append(building)
                    self.building_footprints.append(building_corners)
                    placed_buildings += 1
                    
                    # Add driveway if enabled
                    if params.get('add_driveways', False) and random.random() < 0.7:
                        # Find nearest edge point
                        nearest_distance = float('inf')
                        nearest_edge = None
                        
                        for edge_point in polygon:
                            dist = ((pos_x - edge_point[0])**2 + (pos_y - edge_point[1])**2)**0.5
                            if dist < nearest_distance:
                                nearest_distance = dist
                                nearest_edge = edge_point
                        
                        if nearest_edge:
                            driveway = {
                                'type': 'driveway',
                                'start': (pos_x, pos_y),
                                'end': nearest_edge,
                                'width': random.uniform(2, 4),
                                'zone_id': zone['id']
                            }
                            
                            self.driveways.append(driveway)
                    
                    break
            
            # If we've tried too many times and can't place more buildings, break out
            if attempt == 19:
                break
        
        # Generate single-family houses
        for i in range(num_houses):
            for attempt in range(20):
                # Houses have smaller footprints
                width = random.uniform(avg_house_footprint**0.5 * 0.8, avg_house_footprint**0.5 * 1.2)
                depth = random.uniform(avg_house_footprint**0.5 * 0.8, avg_house_footprint**0.5 * 1.2)
                
                # Random position with regular spacing if enabled
                if params.get('regular_spacing', False):
                    # Create a grid-like arrangement
                    grid_size = int((num_buildings*1.5)**0.5) + 1
                    row = (i + num_apartments) % grid_size
                    col = (i + num_apartments) // grid_size
                    
                    # Position with some jitter
                    pos_x = x + w * (0.1 + 0.8 * row / grid_size) + random.uniform(-w * 0.05, w * 0.05)
                    pos_y = y + h * (0.1 + 0.8 * col / grid_size) + random.uniform(-h * 0.05, h * 0.05)
                else:
                    # Random position
                    pos_x = x + random.uniform(width/2, w - width/2)
                    pos_y = y + random.uniform(depth/2, h - depth/2)
                
                # Random rotation for varied orientations
                rotation = random.uniform(0, 360) if not params.get('regular_spacing', False) else random.choice([0, 90, 180, 270])
                
                # Calculate corners
                rotation_rad = np.radians(rotation)
                c, s = np.cos(rotation_rad), np.sin(rotation_rad)
                R = np.array([[c, -s], [s, c]])
                
                corners_base = np.array([
                    [-width/2, -depth/2],
                    [width/2, -depth/2],
                    [width/2, depth/2],
                    [-width/2, depth/2]
                ])
                
                corners_rotated = np.dot(corners_base, R.T)
                building_corners = [(pos_x + dx, pos_y + dy) for dx, dy in corners_rotated]
                
                # Check if all corners are in zone and not overlapping
                try:
                    building_polygon = Polygon(building_corners)
                    all_corners_in_zone = zone_polygon.contains(building_polygon)
                except:
                    all_corners_in_zone = False
                
                overlap = self.buildings_overlap(building_corners)
                
                if all_corners_in_zone and not overlap:
                    # Houses are shorter
                    min_height, max_height = params['height_range']
                    height = random.uniform(min_height, min_height + (max_height - min_height) * 0.4)
                    
                    # Create house
                    building = {
                        'type': 'house',
                        'zone_type': 'Residential',
                        'zone_id': zone['id'],
                        'position': (pos_x, pos_y),
                        'size': (width, depth),
                        'height': height,
                        'rotation': rotation,
                        'corners': building_corners,
                        'stories': int(height)
                    }
                    
                    self.buildings.append(building)
                    self.building_footprints.append(building_corners)
                    placed_buildings += 1
                    
                    # Add driveway if enabled
                    if params.get('add_driveways', False) and random.random() < 0.9:
                        # Find nearest edge point
                        nearest_distance = float('inf')
                        nearest_edge = None
                        
                        for edge_point in polygon:
                            dist = ((pos_x - edge_point[0])**2 + (pos_y - edge_point[1])**2)**0.5
                            if dist < nearest_distance:
                                nearest_distance = dist
                                nearest_edge = edge_point
                        
                        if nearest_edge:
                            driveway = {
                                'type': 'driveway',
                                'start': (pos_x, pos_y),
                                'end': nearest_edge,
                                'width': random.uniform(1.5, 3),
                                'zone_id': zone['id']
                            }
                            
                            self.driveways.append(driveway)
                    
                    break
            
            # If we've tried too many times and can't place more buildings, break out
            if attempt == 19:
                break
        
        print(f"Placed {placed_buildings} residential buildings in zone {zone['id']}")
        
        # Add community amenities if enabled
        if params.get('add_community_amenities', False) and random.random() < 0.7:
            # Try to place a playground or small park
            for attempt in range(10):
                amenity_width = random.uniform(10, 20)
                amenity_depth = random.uniform(10, 20)
                
                pos_x = x + random.uniform(amenity_width/2, w - amenity_width/2)
                pos_y = y + random.uniform(amenity_depth/2, h - amenity_depth/2)
                
                # Check if position is in zone and not overlapping buildings
                if zone_polygon.contains(Point(pos_x, pos_y)):
                    # Check for overlapping with buildings
                    overlap = False
                    amenity_polygon = Polygon([
                        (pos_x - amenity_width/2, pos_y - amenity_depth/2),
                        (pos_x + amenity_width/2, pos_y - amenity_depth/2),
                        (pos_x + amenity_width/2, pos_y + amenity_depth/2),
                        (pos_x - amenity_width/2, pos_y + amenity_depth/2)
                    ])
                    
                    for building_corners in self.building_footprints:
                        try:
                            building_polygon = Polygon(building_corners)
                            if amenity_polygon.intersects(building_polygon):
                                overlap = True
                                break
                        except:
                            continue
                    
                    if not overlap:
                        # Create community amenity (playground or park)
                        amenity_type = random.choice(['playground', 'small_park'])
                        
                        amenity = {
                            'type': amenity_type,
                            'position': (pos_x, pos_y),
                            'size': (amenity_width, amenity_depth),
                            'zone_id': zone['id']
                        }
                        
                        # Add to plazas list (reusing the data structure)
                        self.plazas.append(amenity)
                        
                        # Add a few trees around amenity
                        num_trees = random.randint(3, 7)
                        for t in range(num_trees):
                            angle = 2 * math.pi * t / num_trees
                            radius = random.uniform(amenity_width*0.6, amenity_width*0.9)
                            
                            tree_x = pos_x + radius * math.cos(angle)
                            tree_y = pos_y + radius * math.sin(angle)
                            
                            if zone_polygon.contains(Point(tree_x, tree_y)):
                                tree = {
                                    'position': (tree_x, tree_y),
                                    'trunk_height': 1.2,
                                    'crown_radius': 1.0,
                                    'tree_height': 2.5,
                                    'zone_id': zone['id']
                                }
                                
                                self.trees.append(tree)
                        
                        break
    
    def _generate_industrial_zone(self, zone):
        """Generate industrial buildings with loading bays and storage yards."""
        polygon = zone['polygon']
        x, y, w, h = zone['bbox']
        params = self.zone_params['Industrial']
        zone_polygon = zone['shapely_polygon']
        
        # Determine building distribution
        area = zone['area']
        plot_coverage = random.uniform(params['plot_coverage'][0], params['plot_coverage'][1])
        total_building_area = area * plot_coverage
        
        # FIXED: Scale better for smaller industrial zones
        max_buildings = min(params['max_buildings'], max(1, int(area / 3000) + 1))
        num_buildings = random.randint(max(1, max_buildings-1), max_buildings)
        
        # Calculate average building footprint
        avg_footprint = total_building_area / num_buildings
        
        # Generate buildings
        placed_buildings = 0
        
        # FIXED: Ensure industrial buildings are definitely placed
        for i in range(num_buildings):
            for attempt in range(30):
                # Industrial buildings are wider
                if params.get('wide_buildings', False):
                    # FIXED: Scale building size based on zone size
                    scale_factor = min(1.0, area / 10000)  # Scale down for small zones
                    width = min(w * 0.6, random.uniform(avg_footprint**0.5 * 1.2, avg_footprint**0.5 * 1.6) * scale_factor)
                    depth = min(h * 0.6, random.uniform(avg_footprint**0.5 * 1.2, avg_footprint**0.5 * 1.6) * scale_factor)
                else:
                    width = min(w * 0.5, random.uniform(avg_footprint**0.5 * 0.8, avg_footprint**0.5 * 1.2))
                    depth = min(h * 0.5, random.uniform(avg_footprint**0.5 * 0.8, avg_footprint**0.5 * 1.2))
                
                # Random position - ensure within bounds
                min_x = x + width/2
                max_x = x + w - width/2
                min_y = y + depth/2
                max_y = y + h - depth/2
                
                # Make sure we have valid bounds
                if min_x >= max_x:
                    min_x, max_x = x, x + w
                if min_y >= max_y:
                    min_y, max_y = y, y + h
                
                pos_x = random.uniform(min_x, max_x)
                pos_y = random.uniform(min_y, max_y)
                
                # Calculate corners
                rotation = random.choice([0, 90, 180, 270])
                rotation_rad = np.radians(rotation)
                c, s = np.cos(rotation_rad), np.sin(rotation_rad)
                R = np.array([[c, -s], [s, c]])
                
                corners_base = np.array([
                    [-width/2, -depth/2],
                    [width/2, -depth/2],
                    [width/2, depth/2],
                    [-width/2, depth/2]
                ])
                
                corners_rotated = np.dot(corners_base, R.T)
                building_corners = [(pos_x + dx, pos_y + dy) for dx, dy in corners_rotated]
                
                # Check if all corners are in zone and not overlapping
                try:
                    building_polygon = Polygon(building_corners)
                    all_corners_in_zone = zone_polygon.contains(building_polygon)
                except:
                    all_corners_in_zone = False
                
                overlap = self.buildings_overlap(building_corners)
                
                if all_corners_in_zone and not overlap:
                    # Industrial buildings are lower height
                    min_height, max_height = params['height_range']
                    height = random.uniform(min_height, max_height)
                    
                    # Create industrial building
                    building = {
                        'type': random.choice(params['building_types']),
                        'zone_type': 'Industrial',
                        'zone_id': zone['id'],
                        'position': (pos_x, pos_y),
                        'size': (width, depth),
                        'height': height,
                        'rotation': rotation,
                        'corners': building_corners,
                        'stories': int(height),
                        'few_windows': True
                    }
                    
                    # Add loading bays if enabled
                    if params.get('add_loading_bays', False) and random.random() < 0.8:
                        # Determine which side will have loading bays
                        bay_side = random.randint(0, 3)
                        num_bays = random.randint(1, 3)
                        building['loading_bays'] = {
                            'side': bay_side,
                            'count': num_bays
                        }
                    
                    # Add smokestacks if enabled
                    if params.get('add_smokestacks', False) and random.random() < 0.6:
                        building['smokestacks'] = random.randint(1, 3)
                    
                    self.buildings.append(building)
                    self.building_footprints.append(building_corners)
                    placed_buildings += 1
                    
                    # Add storage yard next to building if enabled
                    if params.get('add_storage_yards', False) and random.random() < 0.7:
                        # Determine yard size
                        yard_width = width * random.uniform(0.5, 1.0)
                        yard_depth = depth * random.uniform(0.5, 1.0)
                        
                        # Position yard next to building
                        yard_side = random.randint(0, 3)
                        
                        if yard_side == 0:  # North
                            yard_x = pos_x
                            yard_y = pos_y + depth/2 + yard_depth/2 + 2
                        elif yard_side == 1:  # East
                            yard_x = pos_x + width/2 + yard_width/2 + 2
                            yard_y = pos_y
                        elif yard_side == 2:  # South
                            yard_x = pos_x
                            yard_y = pos_y - depth/2 - yard_depth/2 - 2
                        else:  # West
                            yard_x = pos_x - width/2 - yard_width/2 - 2
                            yard_y = pos_y
                        
                        # Check if yard is in zone
                        try:
                            yard_polygon = Polygon([
                                (yard_x - yard_width/2, yard_y - yard_depth/2),
                                (yard_x + yard_width/2, yard_y - yard_depth/2),
                                (yard_x + yard_width/2, yard_y + yard_depth/2),
                                (yard_x - yard_width/2, yard_y + yard_depth/2)
                            ])
                            
                            if zone_polygon.contains(yard_polygon):
                                storage_yard = {
                                    'type': 'storage_yard',
                                    'position': (yard_x, yard_y),
                                    'size': (yard_width, yard_depth),
                                    'rotation': rotation,
                                    'zone_id': zone['id'],
                                    'contains': random.choice(['containers', 'vehicles', 'materials'])
                                }
                                
                                self.storage_yards.append(storage_yard)
                        except:
                            continue
                    
                    break
            
            # If we've tried too many times and can't place more buildings, break out
            if attempt == 29:
                break
        
        print(f"Placed {placed_buildings} industrial buildings in zone {zone['id']}")
        
        # Add security fencing around perimeter if enabled
        if params.get('add_security_fencing', False) and random.random() < 0.8:
            # Create fence along the zone perimeter
            fence = {
                'type': 'security_fence',
                'polygon': polygon,
                'height': random.uniform(1, 2),
                'zone_id': zone['id']
            }
            
            self.security_fences.append(fence)
        
        # Add wide roads if enabled
        if params.get('add_wide_roads', False) and random.random() < 0.7:
            # Create 1-2 wide roads through the zone
            num_roads = random.randint(1, 2)
            
            for r in range(num_roads):
                # Random road direction
                if r == 0:
                    # First road is horizontal or vertical
                    direction = random.choice(['horizontal', 'vertical'])
                else:
                    # Second road is perpendicular to first
                    direction = 'vertical' if direction == 'horizontal' else 'horizontal'
                
                if direction == 'horizontal':
                    # Horizontal road
                    y_pos = y + h * random.uniform(0.3, 0.7)
                    road_points = [(x, y_pos), (x + w, y_pos)]
                else:
                    # Vertical road
                    x_pos = x + w * random.uniform(0.3, 0.7)
                    road_points = [(x_pos, y), (x_pos, y + h)]
                
                road = {
                    'type': 'road',
                    'points': road_points,
                    'width': random.uniform(4, 8),  # Wider roads for industrial
                    'zone_id': zone['id']
                }
                
                self.paths.append(road)
    
    def visualize_2d(self, show_buildings=True, save_path=None):
        """
        Visualize the zones and buildings in 2D.
        
        Args:
            show_buildings: Whether to show generated buildings
            save_path: Path to save the visualization image (optional)
        """
        if self.image is None:
            raise ValueError("No image loaded. Call load_image() first.")
        
        plt.figure(figsize=(12, 12))
        
        # Create a copy of the original image
        img = self.image.copy()
        
        # Draw zone boundaries
        for zone in self.zones:
            color = self.zone_colors[zone['type']]
            cv2.drawContours(img, [zone['contour']], 0, (0, 0, 0), 2)
            
            # Add zone type label
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, f"#{zone.get('id', '')}", (zone['center'][0]-20, zone['center'][1]-10), 
                       font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, zone['type'], (zone['center'][0]-20, zone['center'][1]+10), 
                       font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        if show_buildings:
            # Draw paths and roads first (so they appear under buildings)
            for path in self.paths:
                path_points = path['points']
                width = int(path['width'])
                
                if len(path_points) > 1:
                    for i in range(len(path_points) - 1):
                        pt1 = (int(path_points[i][0]), int(path_points[i][1]))
                        pt2 = (int(path_points[i+1][0]), int(path_points[i+1][1]))
                        cv2.line(img, pt1, pt2, (80, 80, 80), width)
            
            # Draw plazas and courtyards
            for plaza in self.plazas:
                pos_x, pos_y = plaza['position']
                width, depth = plaza['size']
                rotation = plaza.get('rotation', 0)
                
                # Create rectangle
                rect = ((pos_x, pos_y), (width, depth), rotation)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                # Different color based on type
                if plaza.get('type', '') == 'playground':
                    color = (100, 200, 100)  # Light green
                elif plaza.get('type', '') == 'small_park':
                    color = (100, 220, 100)  # Greener
                elif plaza.get('type', '') == 'lawn':
                    color = (150, 230, 150)  # Light green
                else:
                    color = (200, 200, 200)  # Light gray for plazas
                
                cv2.drawContours(img, [box], 0, color, -1)
                cv2.drawContours(img, [box], 0, (100, 100, 100), 1)
            
            # Draw driveways
            for driveway in self.driveways:
                start_x, start_y = driveway['start']
                end_x, end_y = driveway['end']
                width = int(driveway['width'])
                
                cv2.line(img, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (120, 120, 120), width)
            
            # Draw storage yards
            for yard in self.storage_yards:
                pos_x, pos_y = yard['position']
                width, depth = yard['size']
                rotation = yard.get('rotation', 0)
                
                # Create rectangle
                rect = ((pos_x, pos_y), (width, depth), rotation)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                cv2.drawContours(img, [box], 0, (160, 160, 160), -1)  # Gray for yards
                cv2.drawContours(img, [box], 0, (80, 80, 80), 1)
            
            # Draw buildings
            for building in self.buildings:
                pos_x, pos_y = building['position']
                width, depth = building['size']
                rotation = building['rotation']
                
                # Get corners if available, otherwise calculate
                if 'corners' in building:
                    box = np.int0(building['corners'])
                else:
                    # Create rectangle
                    rect = ((pos_x, pos_y), (width, depth), rotation)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                
                # Different color for each building type
                if building['zone_type'] == 'Commercial':
                    color = (230, 76, 60)     # Red
                elif building['zone_type'] == 'Mixed':
                    color = (155, 89, 182)    # Purple
                elif building['zone_type'] == 'Residential':
                    color = (174, 247, 190)   # Light green
                elif building['zone_type'] == 'Industrial':
                    color = (149, 165, 166)   # Gray
                elif building['zone_type'] == 'Civic':
                    color = (243, 156, 18)    # Orange
                else:
                    color = (46, 204, 113)    # Green
                
                cv2.drawContours(img, [box], 0, color, -1)
                cv2.drawContours(img, [box], 0, (0, 0, 0), 1)
                
                # Draw loading bays for industrial buildings
                if 'loading_bays' in building:
                    bay_info = building['loading_bays']
                    bay_side = bay_info['side']
                    num_bays = bay_info['count']
                    
                    # Determine which edge of the building has the loading bays
                    if len(box) >= 4:
                        if bay_side == 0:  # North side
                            bay_edge = (box[0], box[1])
                        elif bay_side == 1:  # East side
                            bay_edge = (box[1], box[2])
                        elif bay_side == 2:  # South side
                            bay_edge = (box[2], box[3])
                        else:  # West side
                            bay_edge = (box[3], box[0])
                        
                        # Draw bays along the edge
                        edge_length = np.linalg.norm(np.array(bay_edge[1]) - np.array(bay_edge[0]))
                        bay_width = edge_length / (num_bays * 2 + 1)
                        
                        for b in range(num_bays):
                            t1 = (b * 2 + 1) / (num_bays * 2)
                            p1 = bay_edge[0] + t1 * (np.array(bay_edge[1]) - np.array(bay_edge[0]))
                            
                            # Draw a yellow rectangle for loading bay
                            bay_rect = ((p1[0], p1[1]), (bay_width, bay_width / 2), rotation)
                            bay_box = cv2.boxPoints(bay_rect)
                            bay_box = np.int0(bay_box)
                            
                            cv2.drawContours(img, [bay_box], 0, (0, 200, 200), -1)
            
            # Draw trees
            for tree in self.trees:
                pos_x, pos_y = tree['position']
                radius = int(tree['crown_radius'] * 5)  # Scale for visibility
                cv2.circle(img, (int(pos_x), int(pos_y)), radius, (0, 150, 0), -1)
            
            # Draw security fences
            for fence in self.security_fences:
                polygon = fence['polygon']
                # Draw a thin line around the perimeter
                for i in range(len(polygon)):
                    pt1 = (int(polygon[i][0]), int(polygon[i][1]))
                    pt2 = (int(polygon[(i+1) % len(polygon)][0]), int(polygon[(i+1) % len(polygon)][1]))
                    cv2.line(img, pt1, pt2, (100, 100, 100), 2)
        
        # Set figure title
        plt.suptitle("Urban Plan with Generated Elements", fontsize=16)
        if self.image_path:
            plt.title(f"Urban Plan: {os.path.basename(self.image_path)}", fontsize=12)
        
        plt.imshow(img)
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"2D visualization saved to {save_path}")
        
        plt.show()
    
    def visualize_3d(self, figsize=(12, 12), elev=20, azim=225, save_path=None):
        """
        Visualize the generated 3D elements.
        
        Args:
            figsize: Figure size as (width, height) tuple
            elev: Elevation angle for 3D view
            azim: Azimuth angle for 3D view
            save_path: Path to save the visualization image (optional)
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Set initial view
        ax.view_init(elev=elev, azim=azim)
        
        # Define ground plane based on image dimensions
        h, w = self.image.shape[:2] if self.image is not None else (1000, 1000)
        max_dim = max(h, w)
        
        # Create ground
        x_min, y_min = 0, 0
        x_max, y_max = 100, 100
        ground_z = 0
        
        # Draw buildings
        for building in self.buildings:
            pos_x, pos_y = building['position']
            width, depth = building['size']
            height = building['height']
            rotation = np.radians(building['rotation'])
            
            # Scale to fit view
            pos_x = pos_x / max_dim * 100
            pos_y = pos_y / max_dim * 100
            width = width / max_dim * 100
            depth = depth / max_dim * 100
            
            # Create rotation matrix
            c, s = np.cos(rotation), np.sin(rotation)
            R = np.array([[c, -s], [s, c]])
            
            # Define corners of the base (before rotation)
            corners_base = np.array([
                [-width/2, -depth/2],
                [width/2, -depth/2],
                [width/2, depth/2],
                [-width/2, depth/2]
            ])
            
            # Apply rotation to corners
            corners_rotated = np.dot(corners_base, R.T)
            
            # Add building center position
            corners_positioned = corners_rotated + np.array([pos_x, pos_y])
            
            # Create 3D coordinates for bottom and top face
            bottom_face = [(x, y, 0) for x, y in corners_positioned]
            top_face = [(x, y, height) for x, y in corners_positioned]
            
            # Create vertices for the building (8 corners)
            vertices = bottom_face + top_face
            
            # Define faces using vertices
            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
                [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
                [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
                [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
                [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
                [vertices[3], vertices[0], vertices[4], vertices[7]]   # left
            ]
            
            # Get building color
            zone_type = building['zone_type']
            color = self.render_colors[zone_type]
            # Add building to plot
            building_collection = Poly3DCollection(faces, alpha=0.9)
            building_collection.set_facecolor(color)
            building_collection.set_edgecolor('black')
            ax.add_collection3d(building_collection)
            
            # Add windows for certain building types
            if zone_type in ['Commercial', 'Mixed', 'Residential', 'Civic'] and height > 3:
                # Only add windows if not explicitly marked as having few windows
                if not building.get('few_windows', False):
                    # Determine window pattern based on building type
                    if zone_type == 'Commercial' or building.get('type', '') == 'office_tower':
                        # Commercial buildings have grid-like windows
                        window_rows = max(1, int(height / 3))
                        window_cols = max(2, int(width / 4))
                        
                        # Add windows to each face except bottom and top
                        for face_idx in range(2, 6):
                            face = faces[face_idx]
                            
                            # Determine face orientation
                            face_width = max(
                                np.linalg.norm(np.array(face[1]) - np.array(face[0])),
                                np.linalg.norm(np.array(face[2]) - np.array(face[3]))
                            )
                            face_height = max(
                                np.linalg.norm(np.array(face[3]) - np.array(face[0])),
                                np.linalg.norm(np.array(face[2]) - np.array(face[1]))
                            )
                            
                            # Calculate window size
                            window_width = face_width / (window_cols * 2)
                            window_height = face_height / (window_rows * 2)
                            
                            # Calculate face center and vectors
                            face_center = np.mean([face[0], face[1], face[2], face[3]], axis=0)
                            
                            # Skip small faces
                            if face_width < 1 or face_height < 1:
                                continue
                            
                            # Create windows as small squares
                            for r in range(window_rows):
                                for c in range(window_cols):
                                    # Position within face
                                    rel_x = (c / (window_cols - 1 or 1) - 0.5) * face_width * 0.8
                                    rel_y = (r / (window_rows - 1 or 1) - 0.5) * face_height * 0.8 + face_height * 0.1
                                    
                                    # Skip windows on the ground floor for mixed use with commercial
                                    if building.get('commercial_ground_floor', False) and r == 0:
                                        continue
                                    
                                    # Create window corners
                                    if face_idx == 2:  # front
                                        window = [
                                            (face_center[0] + rel_x - window_width/2, face_center[1], face_center[2] + rel_y - window_height/2),
                                            (face_center[0] + rel_x + window_width/2, face_center[1], face_center[2] + rel_y - window_height/2),
                                            (face_center[0] + rel_x + window_width/2, face_center[1], face_center[2] + rel_y + window_height/2),
                                            (face_center[0] + rel_x - window_width/2, face_center[1], face_center[2] + rel_y + window_height/2)
                                        ]
                                    elif face_idx == 3:  # right
                                        window = [
                                            (face_center[0], face_center[1] + rel_x - window_width/2, face_center[2] + rel_y - window_height/2),
                                            (face_center[0], face_center[1] + rel_x + window_width/2, face_center[2] + rel_y - window_height/2),
                                            (face_center[0], face_center[1] + rel_x + window_width/2, face_center[2] + rel_y + window_height/2),
                                            (face_center[0], face_center[1] + rel_x - window_width/2, face_center[2] + rel_y + window_height/2)
                                        ]
                                    elif face_idx == 4:  # back
                                        window = [
                                            (face_center[0] - rel_x - window_width/2, face_center[1], face_center[2] + rel_y - window_height/2),
                                            (face_center[0] - rel_x + window_width/2, face_center[1], face_center[2] + rel_y - window_height/2),
                                            (face_center[0] - rel_x + window_width/2, face_center[1], face_center[2] + rel_y + window_height/2),
                                            (face_center[0] - rel_x - window_width/2, face_center[1], face_center[2] + rel_y + window_height/2)
                                        ]
                                    else:  # left
                                        window = [
                                            (face_center[0], face_center[1] - rel_x - window_width/2, face_center[2] + rel_y - window_height/2),
                                            (face_center[0], face_center[1] - rel_x + window_width/2, face_center[2] + rel_y - window_height/2),
                                            (face_center[0], face_center[1] - rel_x + window_width/2, face_center[2] + rel_y + window_height/2),
                                            (face_center[0], face_center[1] - rel_x - window_width/2, face_center[2] + rel_y + window_height/2)
                                        ]
                                    
                                    # Add window
                                    window_collection = Poly3DCollection([window], alpha=0.7)
                                    window_collection.set_facecolor('#add8e6')  # Light blue
                                    window_collection.set_edgecolor('black')
                                    ax.add_collection3d(window_collection)
                    
                    else:  # Residential or mixed
                        # Simpler window pattern
                        window_rows = max(1, int(height / 2))
                        window_cols = max(1, int(width / 4))
                        
                        # Similar window code, but with fewer windows
                        # (Implementation similar to commercial windows but with different parameters)
            
            # Add loading bays for industrial buildings
            if 'loading_bays' in building:
                bay_info = building['loading_bays']
                bay_side = bay_info['side']
                num_bays = bay_info['count']
                
                # Determine which face has loading bays
                bay_face = faces[bay_side + 2]  # +2 to skip bottom and top faces
                
                face_width = max(
                    np.linalg.norm(np.array(bay_face[1]) - np.array(bay_face[0])),
                    np.linalg.norm(np.array(bay_face[2]) - np.array(bay_face[3]))
                )
                
                bay_width = face_width / (num_bays * 2 + 1)
                bay_depth = 0.5
                bay_height = 0.8
                
                for b in range(num_bays):
                    # Position along face
                    rel_pos = (b * 2 + 1) / (num_bays * 2) - 0.5
                    
                    if bay_side == 0:  # front
                        bay_center = (pos_x + rel_pos * width, pos_y - depth/2 - bay_depth/2, bay_height/2)
                        bay_size = (bay_width, bay_depth, bay_height)
                    elif bay_side == 1:  # right
                        bay_center = (pos_x + width/2 + bay_depth/2, pos_y + rel_pos * depth, bay_height/2)
                        bay_size = (bay_depth, bay_width, bay_height)
                    elif bay_side == 2:  # back
                        bay_center = (pos_x + rel_pos * width, pos_y + depth/2 + bay_depth/2, bay_height/2)
                        bay_size = (bay_width, bay_depth, bay_height)
                    else:  # left
                        bay_center = (pos_x - width/2 - bay_depth/2, pos_y + rel_pos * depth, bay_height/2)
                        bay_size = (bay_depth, bay_width, bay_height)
                    
                    # Create loading bay
                    bay_x, bay_y, bay_z = bay_center
                    bay_w, bay_d, bay_h = bay_size
                    
                    # Define corners
                    bay_corners_bottom = [
                        (bay_x - bay_w/2, bay_y - bay_d/2, bay_z - bay_h/2),
                        (bay_x + bay_w/2, bay_y - bay_d/2, bay_z - bay_h/2),
                        (bay_x + bay_w/2, bay_y + bay_d/2, bay_z - bay_h/2),
                        (bay_x - bay_w/2, bay_y + bay_d/2, bay_z - bay_h/2)
                    ]
                    
                    bay_corners_top = [
                        (bay_x - bay_w/2, bay_y - bay_d/2, bay_z + bay_h/2),
                        (bay_x + bay_w/2, bay_y - bay_d/2, bay_z + bay_h/2),
                        (bay_x + bay_w/2, bay_y + bay_d/2, bay_z + bay_h/2),
                        (bay_x - bay_w/2, bay_y + bay_d/2, bay_z + bay_h/2)
                    ]
                    
                    # Create faces
                    bay_faces = []
                    # Bottom
                    bay_faces.append(bay_corners_bottom)
                    # Top
                    bay_faces.append(bay_corners_top)
                    # Sides
                    for i in range(4):
                        side = [
                            bay_corners_bottom[i],
                            bay_corners_bottom[(i+1)%4],
                            bay_corners_top[(i+1)%4],
                            bay_corners_top[i]
                        ]
                        bay_faces.append(side)
                    
                    # Add loading bay
                    bay_collection = Poly3DCollection(bay_faces, alpha=0.8)
                    bay_collection.set_facecolor('#f1c40f')  # Yellow
                    bay_collection.set_edgecolor('black')
                    ax.add_collection3d(bay_collection)
            
            # Add special features based on building type
            if zone_type == 'Industrial' and 'smokestacks' in building:
                for i in range(building['smokestacks']):
                    # Place smokestack on the roof
                    offset_x = width * (random.uniform(-0.3, 0.3))
                    offset_y = depth * (random.uniform(-0.3, 0.3))
                    
                    # Apply rotation to offset
                    rotated_offset = np.dot(np.array([offset_x, offset_y]), R.T)
                    
                    stack_x = pos_x + rotated_offset[0]
                    stack_y = pos_y + rotated_offset[1]
                    stack_height = random.uniform(1, 3)
                    stack_radius = random.uniform(0.3, 0.8)
                    
                    # Create smokestack cylinder
                    stack_bottom = (stack_x, stack_y, height)
                    stack_top = (stack_x, stack_y, height + stack_height)
                    
                    # Create a simplified cylinder
                    cylinder_resolution = 8
                    stack_faces = []
                    
                    # Create bottom and top circles
                    bottom_circle = []
                    top_circle = []
                    
                    for j in range(cylinder_resolution):
                        angle = 2 * math.pi * j / cylinder_resolution
                        x_offset = stack_radius * math.cos(angle)
                        y_offset = stack_radius * math.sin(angle)
                        
                        bottom_circle.append((stack_bottom[0] + x_offset, stack_bottom[1] + y_offset, stack_bottom[2]))
                        top_circle.append((stack_top[0] + x_offset, stack_top[1] + y_offset, stack_top[2]))
                    
                    # Add top and bottom faces
                    stack_faces.append(bottom_circle)
                    stack_faces.append(top_circle)
                    
                    # Add side faces
                    for j in range(cylinder_resolution):
                        side = [
                            bottom_circle[j],
                            bottom_circle[(j+1) % cylinder_resolution],
                            top_circle[(j+1) % cylinder_resolution],
                            top_circle[j]
                        ]
                        stack_faces.append(side)
                    
                    # Add smokestack
                    stack_collection = Poly3DCollection(stack_faces, alpha=0.9)
                    stack_collection.set_facecolor('#7f8c8d')  # Gray
                    stack_collection.set_edgecolor('black')
                    ax.add_collection3d(stack_collection)
        
        # Draw trees
        for tree in self.trees:
            pos_x, pos_y = tree['position']
            trunk_height = tree['trunk_height']
            crown_radius = tree['crown_radius']
            tree_height = tree['tree_height']
            
            # Scale to fit view
            pos_x = pos_x / max_dim * 100
            pos_y = pos_y / max_dim * 100
            
            # Create tree trunk (simplified cylinder)
            trunk_radius = crown_radius * 0.15
            trunk_resolution = 8
            trunk_faces = []
            
            # Create bottom and top circles for trunk
            trunk_bottom_circle = []
            trunk_top_circle = []
            
            for i in range(trunk_resolution):
                angle = 2 * math.pi * i / trunk_resolution
                x_offset = trunk_radius * math.cos(angle)
                y_offset = trunk_radius * math.sin(angle)
                
                trunk_bottom_circle.append((pos_x + x_offset, pos_y + y_offset, ground_z))
                trunk_top_circle.append((pos_x + x_offset, pos_y + y_offset, trunk_height))
            
            # Add trunk faces
            trunk_faces.append(trunk_bottom_circle)
            trunk_faces.append(trunk_top_circle)
            
            for i in range(trunk_resolution):
                side = [
                    trunk_bottom_circle[i],
                    trunk_bottom_circle[(i+1) % trunk_resolution],
                    trunk_top_circle[(i+1) % trunk_resolution],
                    trunk_top_circle[i]
                ]
                trunk_faces.append(side)
            
            # Add trunk
            trunk_collection = Poly3DCollection(trunk_faces, alpha=0.9)
            trunk_collection.set_facecolor('#8B4513')  # Brown
            trunk_collection.set_edgecolor('none')
            ax.add_collection3d(trunk_collection)
            
            # Create tree crown (simplified sphere)
            crown_resolution = 8
            crown_faces = []
            
            # Create points on sphere
            crown_points = []
            for i in range(crown_resolution):
                lat = math.pi * (i / (crown_resolution - 1) - 0.5)
                cos_lat = math.cos(lat)
                sin_lat = math.sin(lat)
                
                lat_points = []
                for j in range(crown_resolution):
                    lon = 2 * math.pi * j / crown_resolution
                    
                    x = pos_x + crown_radius * cos_lat * math.cos(lon)
                    y = pos_y + crown_radius * cos_lat * math.sin(lon)
                    z = trunk_height + crown_radius + crown_radius * sin_lat
                    
                    lat_points.append((x, y, z))
                
                crown_points.append(lat_points)
            
            # Create crown faces
            for i in range(crown_resolution - 1):
                for j in range(crown_resolution):
                    next_j = (j + 1) % crown_resolution
                    
                    face = [
                        crown_points[i][j],
                        crown_points[i][next_j],
                        crown_points[i+1][next_j],
                        crown_points[i+1][j]
                    ]
                    
                    crown_faces.append(face)
            
            # Add crown
            crown_collection = Poly3DCollection(crown_faces, alpha=0.9)
            
            # Color based on tree type
            if tree.get('formal', False):
                crown_collection.set_facecolor('#2ecc71')  # Brighter green for formal trees
            else:
                crown_collection.set_facecolor(self.render_colors['Tree'])
                
            crown_collection.set_edgecolor('none')
            ax.add_collection3d(crown_collection)
        
        # Draw water features
        for feature in self.water_features:
            if feature['type'] == 'water_surface':
                # Create simplified water surface from polygon
                water_polygon = feature['polygon']
                
                # Scale polygon
                scaled_polygon = [(x / max_dim * 100, y / max_dim * 100) for x, y in water_polygon]
                
                # Triangulate the polygon for 3D display
                vertices = np.array(scaled_polygon)
                
                # Create a simple triangulation by creating triangles from the centroid
                centroid_x = np.mean([x for x, y in scaled_polygon])
                centroid_y = np.mean([y for x, y in scaled_polygon])
                
                triangles = []
                for i in range(len(scaled_polygon)):
                    j = (i + 1) % len(scaled_polygon)
                    triangles.append([
                        (scaled_polygon[i][0], scaled_polygon[i][1], ground_z),
                        (scaled_polygon[j][0], scaled_polygon[j][1], ground_z),
                        (centroid_x, centroid_y, ground_z)
                    ])
                
                # Add water surface to plot
                water_collection = Poly3DCollection(triangles, alpha=0.7)
                water_collection.set_facecolor(self.render_colors['Water'])
                water_collection.set_edgecolor('none')
                ax.add_collection3d(water_collection)
                
                # Add waves if enabled
                if feature.get('add_waves', False):
                    # Simplified wave representation
                    wave_count = min(len(scaled_polygon), 10)
                    
                    for w in range(wave_count):
                        # Random position within water area
                        t1 = random.random()
                        t2 = random.random()
                        t3 = 1 - t1 - t2
                        
                        if t3 < 0:
                            t1, t2, t3 = abs(t1), abs(t2), abs(t3)
                            sum_t = t1 + t2 + t3
                            t1, t2, t3 = t1/sum_t, t2/sum_t, t3/sum_t
                        
                        # Random point inside the triangle
                        idx1 = random.randint(0, len(scaled_polygon) - 1)
                        idx2 = (idx1 + 1) % len(scaled_polygon)
                        
                        wave_x = t1 * scaled_polygon[idx1][0] + t2 * scaled_polygon[idx2][0] + t3 * centroid_x
                        wave_y = t1 * scaled_polygon[idx1][1] + t2 * scaled_polygon[idx2][1] + t3 * centroid_y
                        
                        # Create a small ripple
                        wave_radius = random.uniform(0.5, 2.0)
                        wave_resolution = 8
                        
                        wave_circle = []
                        for i in range(wave_resolution):
                            angle = 2 * math.pi * i / wave_resolution
                            x = wave_x + wave_radius * math.cos(angle)
                            y = wave_y + wave_radius * math.sin(angle)
                            wave_circle.append((x, y, ground_z + 0.05))
                        
                        # Add wave
                        wave_collection = Poly3DCollection([wave_circle], alpha=0.3)
                        wave_collection.set_facecolor('#add8e6')  # Light blue
                        wave_collection.set_edgecolor('#add8e6')
                        ax.add_collection3d(wave_collection)
                
            elif feature['type'] == 'pier':
                # Draw pier
                start_x, start_y = feature['start']
                end_x, end_y = feature['end']
                width = feature['width']
                height = feature['height']
                
                # Scale to fit view
                start_x = start_x / max_dim * 100
                start_y = start_y / max_dim * 100
                end_x = end_x / max_dim * 100
                end_y = end_y / max_dim * 100
                width = width / max_dim * 100
                
                # Calculate direction vector
                dir_x = end_x - start_x
                dir_y = end_y - start_y
                dist = (dir_x**2 + dir_y**2)**0.5
                
                if dist > 0:
                    # Normalize direction
                    dir_x /= dist
                    dir_y /= dist
                    
                    # Calculate perpendicular vector
                    perp_x = -dir_y
                    perp_y = dir_x
                    
                    # Calculate corners
                    corners = [
                        (start_x + perp_x * width/2, start_y + perp_y * width/2),
                        (start_x - perp_x * width/2, start_y - perp_y * width/2),
                        (end_x - perp_x * width/2, end_y - perp_y * width/2),
                        (end_x + perp_x * width/2, end_y + perp_y * width/2)
                    ]
                    
                    # Create 3D coordinates
                    bottom_face = [(x, y, ground_z) for x, y in corners]
                    top_face = [(x, y, ground_z + height) for x, y in corners]
                    
                    # Create vertices
                    vertices = bottom_face + top_face
                    
                    # Define faces
                    faces = [
                        [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
                        [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
                        [vertices[0], vertices[1], vertices[5], vertices[4]],  # side 1
                        [vertices[1], vertices[2], vertices[6], vertices[5]],  # side 2
                        [vertices[2], vertices[3], vertices[7], vertices[6]],  # side 3
                        [vertices[3], vertices[0], vertices[4], vertices[7]]   # side 4
                    ]
                    
                    # Add pier to plot
                    pier_collection = Poly3DCollection(faces, alpha=0.9)
                    pier_collection.set_facecolor(self.render_colors['Pier'])
                    pier_collection.set_edgecolor('black')
                    ax.add_collection3d(pier_collection)
            
            elif feature['type'] == 'boat':
                # Draw boat
                pos_x, pos_y = feature['position']
                width, depth = feature['size']
                rotation = feature.get('rotation', 0)
                
                # Scale to fit view
                pos_x = pos_x / max_dim * 100
                pos_y = pos_y / max_dim * 100
                width = width / max_dim * 100
                depth = depth / max_dim * 100
                
                # Create boat shape
                boat_height = 0.5
                
                # Create rotation matrix
                c, s = np.cos(rotation), np.sin(rotation)
                R = np.array([[c, -s], [s, c]])
                
                # Define corners of hull
                hull_base = np.array([
                    [-width/2, -depth/2],
                    [width/2, -depth/2],
                    [width/2, depth/2],
                    [-width/2, depth/2]
                ])
                
                # Make the front more pointed
                hull_base[2] = [width/2 * 0.6, depth/2 * 1.2]
                hull_base[3] = [-width/2 * 0.6, depth/2 * 1.2]
                
                # Apply rotation
                hull_rotated = np.dot(hull_base, R.T)
                
                # Add position offset
                hull_corners = [(pos_x + dx, pos_y + dy) for dx, dy in hull_rotated]
                
                # Create 3D coordinates
                bottom_face = [(x, y, ground_z) for x, y in hull_corners]
                top_face = [(x, y, ground_z + boat_height) for x, y in hull_corners]
                
                # Create boat faces
                boat_faces = []
                # Bottom
                boat_faces.append(bottom_face)
                # Top
                boat_faces.append(top_face)
                # Sides
                for i in range(4):
                    side = [
                        bottom_face[i],
                        bottom_face[(i+1)%4],
                        top_face[(i+1)%4],
                        top_face[i]
                    ]
                    boat_faces.append(side)
                
                # Add boat hull
                hull_collection = Poly3DCollection(boat_faces, alpha=0.9)
                hull_collection.set_facecolor(random.choice(['#3498db', '#f1c40f', '#e74c3c']))
                hull_collection.set_edgecolor('black')
                ax.add_collection3d(hull_collection)
                
                # Add a cabin
                cabin_width = width * 0.5
                cabin_depth = depth * 0.3
                cabin_height = 0.4
                
                # Place cabin slightly toward the back
                cabin_offset_x = -depth * 0.1
                cabin_offset_y = 0
                
                # Rotate offset
                cabin_offset_rotated = np.dot(np.array([cabin_offset_x, cabin_offset_y]), R.T)
                
                cabin_x = pos_x + cabin_offset_rotated[0]
                cabin_y = pos_y + cabin_offset_rotated[1]
                
                # Create cabin corners
                cabin_base = np.array([
                    [-cabin_width/2, -cabin_depth/2],
                    [cabin_width/2, -cabin_depth/2],
                    [cabin_width/2, cabin_depth/2],
                    [-cabin_width/2, cabin_depth/2]
                ])
                
                # Apply same rotation
                cabin_rotated = np.dot(cabin_base, R.T)
                
                # Add position offset
                cabin_corners = [(cabin_x + dx, cabin_y + dy) for dx, dy in cabin_rotated]
                
                # Create 3D coordinates
                cabin_bottom = [(x, y, ground_z + boat_height) for x, y in cabin_corners]
                cabin_top = [(x, y, ground_z + boat_height + cabin_height) for x, y in cabin_corners]
                
                # Create cabin faces
                cabin_faces = []
                # Bottom (not needed as it's inside the boat)
                # Top
                cabin_faces.append(cabin_top)
                # Sides
                for i in range(4):
                    side = [
                        cabin_bottom[i],
                        cabin_bottom[(i+1)%4],
                        cabin_top[(i+1)%4],
                        cabin_top[i]
                    ]
                    cabin_faces.append(side)
                
                # Add cabin
                cabin_collection = Poly3DCollection(cabin_faces, alpha=0.9)
                cabin_collection.set_facecolor('#ecf0f1')  # White
                cabin_collection.set_edgecolor('black')
                ax.add_collection3d(cabin_collection)
            
            elif feature['type'] == 'shoreline':
                # Create a sloped shoreline
                polygon = feature['polygon']
                slope = feature.get('slope', 0.1)
                
                # Scale polygon
                scaled_polygon = [(x / max_dim * 100, y / max_dim * 100) for x, y in polygon]
                
                # Create a slightly smaller inner polygon
                centroid_x = np.mean([x for x, y in scaled_polygon])
                centroid_y = np.mean([y for x, y in scaled_polygon])
                
                inner_polygon = []
                for x, y in scaled_polygon:
                    # Move point slightly toward centroid
                    vector_to_center = (centroid_x - x, centroid_y - y)
                    dist = (vector_to_center[0]**2 + vector_to_center[1]**2)**0.5
                    
                    if dist > 0:
                        unit_vector = (vector_to_center[0] / dist, vector_to_center[1] / dist)
                        inner_x = x + unit_vector[0] * 2
                        inner_y = y + unit_vector[1] * 2
                        inner_polygon.append((inner_x, inner_y))
                    else:
                        inner_polygon.append((x, y))
                
                # Create shore slope faces
                shore_faces = []
                for i in range(len(scaled_polygon)):
                    next_i = (i + 1) % len(scaled_polygon)
                    
                    face = [
                        (scaled_polygon[i][0], scaled_polygon[i][1], ground_z),
                        (scaled_polygon[next_i][0], scaled_polygon[next_i][1], ground_z),
                        (inner_polygon[next_i][0], inner_polygon[next_i][1], ground_z - slope),
                        (inner_polygon[i][0], inner_polygon[i][1], ground_z - slope)
                    ]
                    
                    shore_faces.append(face)
                
                # Add shoreline
                shore_collection = Poly3DCollection(shore_faces, alpha=0.8)
                shore_collection.set_facecolor('#d0d3d4')  # Light gray
                shore_collection.set_edgecolor('none')
                ax.add_collection3d(shore_collection)
        
        # Draw security fences
        for fence in self.security_fences:
            polygon = fence['polygon']
            height = fence.get('height', 1)
            
            # Scale polygon
            scaled_polygon = [(x / max_dim * 100, y / max_dim * 100) for x, y in polygon]
            
            # Create fence faces
            fence_faces = []
            for i in range(len(scaled_polygon)):
                next_i = (i + 1) % len(scaled_polygon)
                
                face = [
                    (scaled_polygon[i][0], scaled_polygon[i][1], ground_z),
                    (scaled_polygon[next_i][0], scaled_polygon[next_i][1], ground_z),
                    (scaled_polygon[next_i][0], scaled_polygon[next_i][1], ground_z + height),
                    (scaled_polygon[i][0], scaled_polygon[i][1], ground_z + height)
                ]
                
                fence_faces.append(face)
            
            # Add fence
            fence_collection = Poly3DCollection(fence_faces, alpha=0.4)
            fence_collection.set_facecolor(self.render_colors['SecurityFence'])
            fence_collection.set_edgecolor('black')
            ax.add_collection3d(fence_collection)
        
        # Set plot limits and labels
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_zlim(0, 30)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Height')
        
        # Set titles
        if self.image_path:
            plt.suptitle(f"3D Urban Plan: {os.path.basename(self.image_path)}", fontsize=16)
        else:
            plt.suptitle('3D Urban Plan Visualization', fontsize=16)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D visualization saved to {save_path}")
        
        plt.show()
    
    def export_to_obj(self, output_path):
        """
        Export the 3D model to OBJ format for external rendering.
        
        Args:
            output_path: Path to save the OBJ file
        """
        print(f"Exporting 3D model to {output_path}...")
        
        with open(output_path, 'w') as f:
            # Write header
            f.write("# Urban 3D Model\n")
            f.write("# Generated with Urban3DGenerator\n\n")
            
            vertex_count = 0
            
            # Export buildings
            for i, building in enumerate(self.buildings):
                f.write(f"o Building_{i}\n")
                
                pos_x, pos_y = building['position']
                width, depth = building['size']
                height = building['height']
                rotation = np.radians(building['rotation'])
                
                # Create rotation matrix
                c, s = np.cos(rotation), np.sin(rotation)
                R = np.array([[c, -s], [s, c]])
                
                # Define corners of the base (before rotation)
                corners_base = np.array([
                    [-width/2, -depth/2],
                    [width/2, -depth/2],
                    [width/2, depth/2],
                    [-width/2, depth/2]
                ])
                
                # Apply rotation to corners
                corners_rotated = np.dot(corners_base, R.T)
                
                # Add building center position
                corners_positioned = corners_rotated + np.array([pos_x, pos_y])
                
                # Write vertices
                for x, y in corners_positioned:
                    f.write(f"v {x} {y} 0\n")  # Bottom vertices
                
                for x, y in corners_positioned:
                    f.write(f"v {x} {y} {height}\n")  # Top vertices
                
                # Write faces (adjust indices for OBJ 1-based indexing)
                f.write(f"f {vertex_count+1} {vertex_count+2} {vertex_count+3} {vertex_count+4}\n")  # Bottom
                f.write(f"f {vertex_count+5} {vertex_count+6} {vertex_count+7} {vertex_count+8}\n")  # Top
                f.write(f"f {vertex_count+1} {vertex_count+2} {vertex_count+6} {vertex_count+5}\n")  # Side 1
                f.write(f"f {vertex_count+2} {vertex_count+3} {vertex_count+7} {vertex_count+6}\n")  # Side 2
                f.write(f"f {vertex_count+3} {vertex_count+4} {vertex_count+8} {vertex_count+7}\n")  # Side 3
                f.write(f"f {vertex_count+4} {vertex_count+1} {vertex_count+5} {vertex_count+8}\n")  # Side 4
                
                vertex_count += 8
                
                # Add special features (like smokestacks) for industrial buildings
                if building['zone_type'] == 'Industrial' and 'smokestacks' in building:
                    for j in range(building['smokestacks']):
                        # Place smokestack on the roof
                        offset_x = width * (random.uniform(-0.3, 0.3))
                        offset_y = depth * (random.uniform(-0.3, 0.3))
                        
                        # Apply rotation to offset
                        rotated_offset = np.dot(np.array([offset_x, offset_y]), R.T)
                        
                        stack_x = pos_x + rotated_offset[0]
                        stack_y = pos_y + rotated_offset[1]
                        stack_height = random.uniform(1, 3)
                        stack_radius = random.uniform(0.3, 0.8)
                        
                        # Create smokestack vertices (simplified cylinder)
                        stack_resolution = 8
                        
                        f.write(f"o Smokestack_{i}_{j}\n")
                        
                        # Create bottom circle
                        bottom_vertices = []
                        for k in range(stack_resolution):
                            angle = 2 * math.pi * k / stack_resolution
                            x = stack_x + stack_radius * math.cos(angle)
                            y = stack_y + stack_radius * math.sin(angle)
                            f.write(f"v {x} {y} {height}\n")
                            bottom_vertices.append(vertex_count + k + 1)
                        
                        # Create top circle
                        top_vertices = []
                        for k in range(stack_resolution):
                            angle = 2 * math.pi * k / stack_resolution
                            x = stack_x + stack_radius * math.cos(angle)
                            y = stack_y + stack_radius * math.sin(angle)
                            f.write(f"v {x} {y} {height + stack_height}\n")
                            top_vertices.append(vertex_count + stack_resolution + k + 1)
                        
                        # Write faces
                        # Bottom
                        f.write(f"f")
                        for v in bottom_vertices:
                            f.write(f" {v}")
                        f.write("\n")
                        
                        # Top
                        f.write(f"f")
                        for v in reversed(top_vertices):  # Reverse to maintain correct winding
                            f.write(f" {v}")
                        f.write("\n")
                        
                        # Sides
                        for k in range(stack_resolution):
                            next_k = (k + 1) % stack_resolution
                            f.write(f"f {bottom_vertices[k]} {bottom_vertices[next_k]} {top_vertices[next_k]} {top_vertices[k]}\n")
                        
                        vertex_count += stack_resolution * 2
            
            # Export trees (simplified)
            for i, tree in enumerate(self.trees):
                pos_x, pos_y = tree['position']
                trunk_height = tree['trunk_height']
                crown_radius = tree['crown_radius']
                
                # Create simplified tree (cone and cylinder)
                trunk_radius = crown_radius * 0.15
                trunk_resolution = 8
                
                f.write(f"o Tree_{i}_trunk\n")
                
                # Create trunk
                # Bottom circle
                trunk_bottom = []
                for j in range(trunk_resolution):
                    angle = 2 * math.pi * j / trunk_resolution
                    x = pos_x + trunk_radius * math.cos(angle)
                    y = pos_y + trunk_radius * math.sin(angle)
                    f.write(f"v {x} {y} 0\n")
                    trunk_bottom.append(vertex_count + j + 1)
                
                # Top circle
                trunk_top = []
                for j in range(trunk_resolution):
                    angle = 2 * math.pi * j / trunk_resolution
                    x = pos_x + trunk_radius * math.cos(angle)
                    y = pos_y + trunk_radius * math.sin(angle)
                    f.write(f"v {x} {y} {trunk_height}\n")
                    trunk_top.append(vertex_count + trunk_resolution + j + 1)
                
                # Write trunk faces
                # Bottom
                f.write(f"f")
                for v in trunk_bottom:
                    f.write(f" {v}")
                f.write("\n")
                
                # Top
                f.write(f"f")
                for v in reversed(trunk_top):
                    f.write(f" {v}")
                f.write("\n")
                
                # Sides
                for j in range(trunk_resolution):
                    next_j = (j + 1) % trunk_resolution
                    f.write(f"f {trunk_bottom[j]} {trunk_bottom[next_j]} {trunk_top[next_j]} {trunk_top[j]}\n")
                
                vertex_count += trunk_resolution * 2
                
                # Create crown (cone)
                crown_resolution = 8
                
                f.write(f"o Tree_{i}_crown\n")
                
                # Create base circle
                crown_base = []
                for j in range(crown_resolution):
                    angle = 2 * math.pi * j / crown_resolution
                    x = pos_x + crown_radius * math.cos(angle)
                    y = pos_y + crown_radius * math.sin(angle)
                    f.write(f"v {x} {y} {trunk_height}\n")
                    crown_base.append(vertex_count + j + 1)
                
                # Create tip
                f.write(f"v {pos_x} {pos_y} {trunk_height + crown_radius * 2}\n")
                tip_vertex = vertex_count + crown_resolution + 1
                
                # Write crown faces
                # Base
                f.write(f"f")
                for v in crown_base:
                    f.write(f" {v}")
                f.write("\n")
                
                # Sides
                for j in range(crown_resolution):
                    next_j = (j + 1) % crown_resolution
                    f.write(f"f {crown_base[j]} {crown_base[next_j]} {tip_vertex}\n")
                
                vertex_count += crown_resolution + 1
            
            # Export water features (simplified)
            for i, feature in enumerate(self.water_features):
                if feature['type'] == 'water_surface':
                    water_polygon = feature['polygon']
                    
                    f.write(f"o Water_{i}\n")
                    
                    # Write vertices
                    start_vertex = vertex_count + 1
                    for x, y in water_polygon:
                        f.write(f"v {x} {y} 0\n")
                    
                    # Add centroid
                    centroid_x = sum(x for x, y in water_polygon) / len(water_polygon)
                    centroid_y = sum(y for x, y in water_polygon) / len(water_polygon)
                    f.write(f"v {centroid_x} {centroid_y} 0\n")
                    
                    # Write faces (triangulate from centroid)
                    centroid_vertex = start_vertex + len(water_polygon)
                    for j in range(len(water_polygon)):
                        v1 = start_vertex + j
                        v2 = start_vertex + (j + 1) % len(water_polygon)
                        f.write(f"f {v1} {v2} {centroid_vertex}\n")
                    
                    vertex_count += len(water_polygon) + 1
                
                elif feature['type'] == 'pier':
                    start_x, start_y = feature['start']
                    end_x, end_y = feature['end']
                    width = feature['width']
                    height = feature['height']
                    
                    f.write(f"o Pier_{i}\n")
                    
                    # Calculate direction and perpendicular vectors
                    dir_x = end_x - start_x
                    dir_y = end_y - start_y
                    dist = (dir_x**2 + dir_y**2)**0.5
                    
                    if dist > 0:
                        dir_x /= dist
                        dir_y /= dist
                        
                        perp_x = -dir_y
                        perp_y = dir_x
                        
                        # Calculate corners
                        corners = [
                            (start_x + perp_x * width/2, start_y + perp_y * width/2),
                            (start_x - perp_x * width/2, start_y - perp_y * width/2),
                            (end_x - perp_x * width/2, end_y - perp_y * width/2),
                            (end_x + perp_x * width/2, end_y + perp_y * width/2)
                        ]
                        
                        # Write vertices
                        for x, y in corners:
                            f.write(f"v {x} {y} 0\n")  # Bottom
                        
                        for x, y in corners:
                            f.write(f"v {x} {y} {height}\n")  # Top
                        
                        # Write faces
                        f.write(f"f {vertex_count+1} {vertex_count+2} {vertex_count+3} {vertex_count+4}\n")  # Bottom
                        f.write(f"f {vertex_count+5} {vertex_count+6} {vertex_count+7} {vertex_count+8}\n")  # Top
                        f.write(f"f {vertex_count+1} {vertex_count+2} {vertex_count+6} {vertex_count+5}\n")  # Side 1
                        f.write(f"f {vertex_count+2} {vertex_count+3} {vertex_count+7} {vertex_count+6}\n")  # Side 2
                        f.write(f"f {vertex_count+3} {vertex_count+4} {vertex_count+8} {vertex_count+7}\n")  # Side 3
                        f.write(f"f {vertex_count+4} {vertex_count+1} {vertex_count+5} {vertex_count+8}\n")  # Side 4
                        
                        vertex_count += 8
        
        print(f"3D model exported to {output_path}")
        return output_path


def run_urban_generator(image_path, output_dir="./output", export_obj=True, show_vis=True):
    """
    Run the urban generator with the given image path.
    
    Args:
        image_path: Path to the voronoi diagram image
        output_dir: Directory to save output files
        export_obj: Whether to export 3D model as OBJ
        show_vis: Whether to show visualizations
    
    Returns:
        Dictionary with paths to generated files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up output paths
    basename = os.path.splitext(os.path.basename(image_path))[0]
    vis_2d_path = os.path.join(output_dir, f"{basename}_2d.png")
    vis_3d_path = os.path.join(output_dir, f"{basename}_3d.png")
    obj_path = os.path.join(output_dir, f"{basename}.obj")
    
    try:
        # Create generator
        generator = Urban3DGenerator(image_path)
        
        # Extract zones
        generator.extract_zones()
        
        # Generate buildings
        generator.generate_buildings()
        
        # Visualize 2D
        if show_vis:
            generator.visualize_2d(save_path=vis_2d_path, show_buildings=True)
        else:
            # Save without showing
            generator.visualize_2d(save_path=vis_2d_path, show_buildings=True)
            plt.close()
        
        # Visualize 3D
        if show_vis:
            generator.visualize_3d(save_path=vis_3d_path)
        else:
            # Save without showing
            generator.visualize_3d(save_path=vis_3d_path)
            plt.close()
        
        # Export OBJ if requested
        if export_obj:
            generator.export_to_obj(obj_path)
        
        return {
            "vis_2d": vis_2d_path,
            "vis_3d": vis_3d_path,
            "obj": obj_path if export_obj else None
        }
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Set the path to your voronoi diagram image here
    IMAGE_PATH = r"d:\GENAI\UDGAN Datasets-20250415T062915Z-003\UDGAN Datasets\voronoi diagrams images dataset\Batch_1_766.png"  # Update this to your actual image path
    
    # Run the generator
    output_files = run_urban_generator(
        image_path=IMAGE_PATH,
        output_dir="./urban_output",
        export_obj=True,
        show_vis=True
    )
    
    if output_files:
        print("\nGeneration complete!")
        print(f"2D Visualization: {output_files['vis_2d']}")
        print(f"3D Visualization: {output_files['vis_3d']}")
        if output_files['obj']:
            print(f"3D Model (OBJ): {output_files['obj']}")
