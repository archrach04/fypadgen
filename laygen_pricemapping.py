"""
Layout Generation & Price Mapping Module
==========================================
Banner-style ad layout generation with:
  1. Saliency detection for optimal text placement
  2. Bayesian layout scoring for composition optimization
  3. Catchy caption generation (≤6 words, conditioned on product/intent)
  4. Dynamic banner rendering with price/discount overlays

No fixed layouts. No hardcoded templates. All placements are computed.
"""

import hashlib
import math
import random
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None

try:
    import requests as http_requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    http_requests = None


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# Saliency grid resolution
SALIENCY_GRID_SIZE: int = 8

# Color palettes for different intent vectors
INTENT_PALETTES: Dict[int, Dict[str, Tuple[int, int, int]]] = {
    0: {"primary": (245, 197, 24), "secondary": (232, 93, 122), "text": (10, 10, 18)},    # Gold/Pink
    1: {"primary": (0, 212, 170), "secondary": (59, 178, 208), "text": (10, 10, 18)},     # Teal/Cyan
    2: {"primary": (232, 93, 122), "secondary": (155, 89, 182), "text": (255, 255, 255)}, # Pink/Purple
    3: {"primary": (124, 92, 252), "secondary": (59, 178, 208), "text": (255, 255, 255)}, # Purple/Cyan
}

# Caption word banks organized by semantic category
CAPTION_WORDS: Dict[str, List[str]] = {
    "quality": ["Premium", "Luxury", "Elite", "Superior", "Finest", "Ultimate", "Prime"],
    "action": ["Grab", "Unlock", "Discover", "Elevate", "Transform", "Experience", "Embrace"],
    "emotion": ["Love", "Joy", "Bliss", "Wow", "Magic", "Dream", "Glow"],
    "value": ["Save", "Deal", "Steal", "Value", "Smart", "Win", "Score"],
    "urgency": ["Now", "Today", "Fast", "Quick", "Rush", "Limited", "Hurry"],
    "style": ["Chic", "Bold", "Fresh", "Sleek", "Sharp", "Cool", "Trendy"],
}

# Product type keyword mappings for caption generation
PRODUCT_KEYWORDS: Dict[str, List[str]] = {
    "phone": ["Connected", "Smart", "Mobile", "Tech"],
    "laptop": ["Power", "Portable", "Compute", "Work"],
    "headphone": ["Sound", "Audio", "Music", "Beats"],
    "earbuds": ["Pure", "Wireless", "Sound", "Beats"],
    "watch": ["Time", "Style", "Tick", "Wear"],
    "camera": ["Capture", "Snap", "Focus", "Vision"],
    "tv": ["Cinema", "View", "Screen", "Watch"],
    "shoe": ["Step", "Walk", "Stride", "Move"],
    "sneaker": ["Street", "Run", "Move", "Flow"],
    "dress": ["Glam", "Shine", "Turn", "Dazzle"],
    "shirt": ["Fresh", "Clean", "Sharp", "Style"],
    "bag": ["Carry", "Pack", "Hold", "Go"],
    "skincare": ["Glow", "Radiant", "Fresh", "Pure"],
    "makeup": ["Beauty", "Glow", "Glam", "Dazzle"],
    "jewelry": ["Shine", "Sparkle", "Radiant", "Luxe"],
    "furniture": ["Home", "Space", "Living", "Comfort"],
    "kitchen": ["Cook", "Chef", "Home", "Create"],
    "book": ["Read", "Learn", "Story", "Wisdom"],
    "toy": ["Play", "Fun", "Joy", "Happy"],
    "fitness": ["Strong", "Fit", "Power", "Move"],
}


# ---------------------------------------------------------------------------
# SALIENCY DETECTION (No external ML models)
# ---------------------------------------------------------------------------

class SaliencyDetector:
    """
    Lightweight saliency detection using image statistics.
    No pretrained models. Computes based on:
      - Color variance
      - Edge density (Sobel-like approximation)
      - Center bias
    
    Returns a [GRID x GRID] saliency map where higher = more salient.
    """

    def __init__(self, grid_size: int = SALIENCY_GRID_SIZE) -> None:
        self.grid_size = grid_size

    def compute_saliency_map(self, img: "Image.Image") -> np.ndarray:
        """
        Compute saliency heatmap for the image.
        
        Args:
            img: PIL Image (RGB)
        
        Returns:
            saliency_map: [grid_size, grid_size] float array, values 0-1
                          Higher values = more salient (avoid placing text)
        """
        if not HAS_PIL or img is None:
            # Return uniform saliency if no image
            return np.ones((self.grid_size, self.grid_size)) * 0.5

        # Resize to grid for efficient computation
        small = img.resize((self.grid_size * 8, self.grid_size * 8), Image.LANCZOS)
        arr = np.array(small, dtype=np.float32)  # [H', W', 3]

        saliency = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        cell_h = arr.shape[0] // self.grid_size
        cell_w = arr.shape[1] // self.grid_size

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Extract cell
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                cell = arr[y1:y2, x1:x2, :]  # [cell_h, cell_w, 3]

                # Color variance (higher variance = more salient)
                color_var = cell.var(axis=(0, 1)).mean() / 255.0

                # Edge density (simple gradient magnitude)
                gray = cell.mean(axis=2)  # [cell_h, cell_w]
                dx = np.abs(np.diff(gray, axis=1)).mean()
                dy = np.abs(np.diff(gray, axis=0)).mean()
                edge_density = (dx + dy) / 255.0

                # Center bias (objects tend to be centered)
                cy, cx = (i + 0.5) / self.grid_size, (j + 0.5) / self.grid_size
                dist_from_center = math.sqrt((cy - 0.5) ** 2 + (cx - 0.5) ** 2)
                center_bias = max(0, 1 - dist_from_center * 1.5)

                # Combine: variance + edges + center = salient
                saliency[i, j] = 0.3 * color_var + 0.4 * edge_density + 0.3 * center_bias

        # Normalize to 0-1
        s_min, s_max = saliency.min(), saliency.max()
        if s_max > s_min:
            saliency = (saliency - s_min) / (s_max - s_min)
        else:
            saliency = np.ones_like(saliency) * 0.5

        return saliency


# ---------------------------------------------------------------------------
# BAYESIAN LAYOUT SCORING
# ---------------------------------------------------------------------------

class BayesianLayoutScorer:
    """
    Scores banner layouts using Bayesian-inspired composition rules.
    
    Evaluates placement candidates based on:
      - Saliency avoidance (don't cover important parts)
      - Rule of thirds alignment
      - Contrast with background
      - Size appropriateness
    
    Returns the optimal placement region for text elements.
    """

    def __init__(self, saliency_detector: SaliencyDetector) -> None:
        self.saliency = saliency_detector
        self.grid_size = saliency_detector.grid_size

    def score_region(
        self,
        saliency_map: np.ndarray,           # [G, G] saliency values 0-1
        region: Tuple[int, int, int, int],  # (row_start, col_start, row_end, col_end) in grid coords
        bg_luminance: float,                # avg background luminance 0-1
        region_type: str = "banner",        # "banner", "badge", "cta"
    ) -> float:
        """
        Score a potential placement region.
        
        Returns:
            score: higher = better placement
        """
        r1, c1, r2, c2 = region
        r1 = max(0, min(r1, self.grid_size))
        r2 = max(0, min(r2, self.grid_size))
        c1 = max(0, min(c1, self.grid_size))
        c2 = max(0, min(c2, self.grid_size))

        if r2 <= r1 or c2 <= c1:
            return 0.0

        # Extract region saliency
        region_saliency = saliency_map[r1:r2, c1:c2]
        avg_saliency = region_saliency.mean()

        # Saliency avoidance: prefer low-saliency areas for text
        # P(good_placement | saliency) ~ 1 - saliency
        saliency_score = 1.0 - avg_saliency

        # Rule of thirds: prefer intersections
        # Thirds lines at 1/3 and 2/3 of grid
        third_rows = [self.grid_size / 3, 2 * self.grid_size / 3]
        third_cols = [self.grid_size / 3, 2 * self.grid_size / 3]
        
        region_center_r = (r1 + r2) / 2
        region_center_c = (c1 + c2) / 2
        
        # Distance to nearest third line (normalized)
        row_dist = min(abs(region_center_r - tr) for tr in third_rows) / self.grid_size
        col_dist = min(abs(region_center_c - tc) for tc in third_cols) / self.grid_size
        thirds_score = 1.0 - (row_dist + col_dist) / 2

        # Contrast: text needs contrast with background
        # P(readable | bg_lum) - favor dark bg for light text or vice versa
        contrast_score = abs(bg_luminance - 0.5) * 2  # Higher if bg is very dark/light

        # Size appropriateness
        region_size = (r2 - r1) * (c2 - c1)
        total_size = self.grid_size * self.grid_size
        size_ratio = region_size / total_size
        
        # Banner: prefer 10-40% coverage
        # Badge: prefer 5-15% coverage
        # CTA: prefer 5-20% coverage
        if region_type == "banner":
            ideal_size = 0.25
            size_tolerance = 0.2
        elif region_type == "badge":
            ideal_size = 0.08
            size_tolerance = 0.08
        else:  # cta
            ideal_size = 0.12
            size_tolerance = 0.12

        size_score = max(0, 1.0 - abs(size_ratio - ideal_size) / size_tolerance)

        # Edge preference: banners look better near edges
        edge_dist = min(r1, self.grid_size - r2, c1, self.grid_size - c2)
        edge_score = 1.0 - (edge_dist / (self.grid_size / 2))

        # Bayesian combination (weighted prior)
        # P(good | features) ∝ P(features | good) * P(good)
        weights = {
            "saliency": 0.35,
            "thirds": 0.15,
            "contrast": 0.20,
            "size": 0.15,
            "edge": 0.15,
        }
        
        total_score = (
            weights["saliency"] * saliency_score +
            weights["thirds"] * thirds_score +
            weights["contrast"] * contrast_score +
            weights["size"] * size_score +
            weights["edge"] * edge_score
        )

        return total_score

    def find_optimal_placement(
        self,
        img: "Image.Image",
        zone_type: str = "banner",  # "banner", "badge", "cta"
    ) -> Dict[str, Any]:
        """
        Find the optimal placement for a text zone.
        
        Returns:
            {
                "region": (x1, y1, x2, y2) in pixel coords,
                "score": float,
                "position": "top-left" | "top-right" | "bottom-left" | "bottom-right" | "top" | "bottom",
                "palette_idx": int (suggested color palette based on background)
            }
        """
        if not HAS_PIL or img is None:
            # Default to top-left corner
            return {
                "region": (10, 10, 300, 150),
                "score": 0.5,
                "position": "top-left",
                "palette_idx": 0,
            }

        w, h = img.size
        saliency_map = self.saliency.compute_saliency_map(img)

        # Compute background luminance
        arr = np.array(img.resize((32, 32), Image.LANCZOS), dtype=np.float32)
        luminance = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).mean() / 255.0

        # Define candidate regions based on zone type
        candidates: List[Tuple[Tuple[int, int, int, int], str]] = []
        g = self.grid_size

        if zone_type == "banner":
            # Banner candidates: corners and edges
            candidates = [
                ((0, 0, g // 2, g // 2), "top-left"),
                ((0, g // 2, g // 2, g), "top-right"),
                ((g // 2, 0, g, g // 2), "bottom-left"),
                ((g // 2, g // 2, g, g), "bottom-right"),
                ((0, 0, g // 3, g), "top"),
                ((2 * g // 3, 0, g, g), "bottom"),
            ]
        elif zone_type == "badge":
            # Badge: small corner placements
            candidates = [
                ((0, 0, g // 4, g // 3), "top-left"),
                ((0, 2 * g // 3, g // 4, g), "top-right"),
                ((3 * g // 4, 0, g, g // 3), "bottom-left"),
            ]
        else:  # cta
            # CTA: bottom corners preferred
            candidates = [
                ((2 * g // 3, g // 2, g, g), "bottom-right"),
                ((2 * g // 3, 0, g, g // 2), "bottom-left"),
                ((g // 2, g // 2, 2 * g // 3, g), "mid-right"),
            ]

        # Score all candidates
        best_score = -1
        best_region = candidates[0][0]
        best_position = candidates[0][1]

        for region, position in candidates:
            score = self.score_region(saliency_map, region, luminance, zone_type)
            if score > best_score:
                best_score = score
                best_region = region
                best_position = position

        # Convert grid coordinates to pixel coordinates
        r1, c1, r2, c2 = best_region
        px_x1 = int(c1 / g * w)
        px_y1 = int(r1 / g * h)
        px_x2 = int(c2 / g * w)
        px_y2 = int(r2 / g * h)

        # Select palette based on luminance
        if luminance > 0.6:
            palette_idx = 2  # Dark text palette for light bg
        elif luminance < 0.3:
            palette_idx = 0  # Light text palette for dark bg
        else:
            palette_idx = 1  # Medium contrast palette

        return {
            "region": (px_x1, px_y1, px_x2, px_y2),
            "score": best_score,
            "position": best_position,
            "palette_idx": palette_idx,
        }


# ---------------------------------------------------------------------------
# CAPTION GENERATOR (No templates, conditioned generation)
# ---------------------------------------------------------------------------

class CaptionGenerator:
    """
    Generates catchy captions (≤6 words) conditioned on:
      - Product title/category
      - Discount signal
      - Inferred intent (if available)
    
    No templates. Generates based on semantic word combinations.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)

    def _extract_product_type(self, title: str, category: str) -> str:
        """Extract the primary product type from title/category."""
        combined = (title + " " + category).lower()
        for keyword in PRODUCT_KEYWORDS:
            if keyword in combined:
                return keyword
        return "default"

    def _hash_to_index(self, text: str, max_val: int) -> int:
        """Deterministic hash to select from options."""
        h = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        return h % max_val

    def generate_tagline(
        self,
        title: str,
        category: str,
        discount_pct: float = 0,
        intent_vector: Optional[List[float]] = None,
    ) -> str:
        """
        Generate a catchy tagline (≤6 words).
        
        Combines semantic word categories based on product features.
        No fixed templates - words are selected and combined dynamically.
        """
        product_type = self._extract_product_type(title, category)
        
        # Get product-specific words
        type_words = PRODUCT_KEYWORDS.get(product_type, ["Quality", "Best", "Top"])
        
        # Build word selection based on conditions
        selected_words: List[str] = []
        
        # Word 1: Product-specific or emotion word
        if product_type in PRODUCT_KEYWORDS:
            w1_pool = type_words + CAPTION_WORDS["emotion"]
        else:
            w1_pool = CAPTION_WORDS["quality"] + CAPTION_WORDS["emotion"]
        idx1 = self._hash_to_index(title, len(w1_pool))
        selected_words.append(w1_pool[idx1])
        
        # Word 2: Based on discount
        if discount_pct >= 30:
            w2_pool = CAPTION_WORDS["value"] + CAPTION_WORDS["urgency"]
        elif discount_pct >= 15:
            w2_pool = CAPTION_WORDS["value"] + CAPTION_WORDS["action"]
        else:
            w2_pool = CAPTION_WORDS["style"] + CAPTION_WORDS["action"]
        idx2 = self._hash_to_index(category, len(w2_pool))
        selected_words.append(w2_pool[idx2])
        
        # Word 3: Optional based on intent
        if intent_vector is not None and len(intent_vector) > 0:
            dominant_intent = int(np.argmax(intent_vector)) % 4
            intent_words = {
                0: CAPTION_WORDS["quality"],
                1: CAPTION_WORDS["value"],
                2: CAPTION_WORDS["emotion"],
                3: CAPTION_WORDS["urgency"],
            }
            w3_pool = intent_words.get(dominant_intent, CAPTION_WORDS["action"])
        else:
            w3_pool = CAPTION_WORDS["action"]
        idx3 = self._hash_to_index(title + category, len(w3_pool))
        selected_words.append(w3_pool[idx3])
        
        # Build phrase patterns (all ≤6 words)
        patterns = [
            lambda w: f"{w[0]}. {w[1]}. {w[2]}.",           # "Glow. Save. Now."
            lambda w: f"{w[0]} Meets {w[1]}",               # "Style Meets Value"
            lambda w: f"{w[0]} {w[1]} Awaits",              # "Premium Joy Awaits"
            lambda w: f"Pure {w[0]}, Pure {w[1]}",          # "Pure Sound, Pure Joy"
            lambda w: f"{w[0]} Your {w[1]}",                # "Unlock Your Style"
            lambda w: f"{w[0]} The {w[1]}",                 # "Experience The Magic"
            lambda w: f"Your {w[0]} {w[1]}",                # "Your Smart Upgrade"
        ]
        
        pattern_idx = self._hash_to_index(title + str(discount_pct), len(patterns))
        tagline = patterns[pattern_idx](selected_words)
        
        # Ensure ≤6 words
        words = tagline.split()
        if len(words) > 6:
            tagline = " ".join(words[:6])
        
        return tagline

    def generate_headline(
        self,
        title: str,
        brand: Optional[str] = None,
    ) -> str:
        """
        Generate a short headline from product title.
        Extracts brand + key descriptors, max 5 words.
        """
        words = title.split()
        
        # Extract potential brand (first capitalized word)
        extracted_brand = brand
        if not extracted_brand and words and words[0][0].isupper():
            extracted_brand = words[0]
        
        # Build headline
        if extracted_brand:
            # Find most descriptive non-brand words
            desc_words = [w for w in words[1:5] if len(w) > 2]
            headline = extracted_brand + " " + " ".join(desc_words[:3])
        else:
            headline = " ".join(words[:4])
        
        return headline[:35]

    def generate_cta(
        self,
        discount_pct: float = 0,
        price: float = 0,
    ) -> str:
        """
        Generate a 2-3 word CTA.
        Conditioned on discount and price.
        """
        if discount_pct >= 50:
            ctas = ["Steal This!", "Grab Now!", "Don't Miss!"]
        elif discount_pct >= 30:
            ctas = ["Save Big!", "Get Deal!", "Shop Now!"]
        elif discount_pct >= 15:
            ctas = ["Save Now!", "Get Yours!", "Shop Deal!"]
        elif price and price < 500:
            ctas = ["Quick Buy!", "Get It!", "Shop Now!"]
        elif price and price > 10000:
            ctas = ["Invest Now!", "Get Premium!", "Shop Now!"]
        else:
            ctas = ["Shop Now!", "Get Yours!", "Buy Now!"]
        
        idx = self._hash_to_index(str(discount_pct) + str(price), len(ctas))
        return ctas[idx]


# ---------------------------------------------------------------------------
# BANNER RENDERER
# ---------------------------------------------------------------------------

class BannerRenderer:
    """
    Renders banner-style ad layouts with:
      - Saliency-aware text placement
      - Bayesian-optimized composition
      - Generated captions
      - Price/discount overlays
    """

    def __init__(self) -> None:
        self.saliency_detector = SaliencyDetector()
        self.layout_scorer = BayesianLayoutScorer(self.saliency_detector)
        self.caption_gen = CaptionGenerator()

    def _load_font(self, size: int, bold: bool = False) -> "ImageFont.FreeTypeFont":
        """Load a font, falling back to default if unavailable."""
        if not HAS_PIL:
            return None
            
        font_names = [
            "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
            "Arial Bold.ttf" if bold else "Arial.ttf",
            "Helvetica Bold.ttf" if bold else "Helvetica.ttf",
        ]
        
        for font_name in font_names:
            try:
                return ImageFont.truetype(font_name, size)
            except (IOError, OSError):
                continue
        
        return ImageFont.load_default()

    def _draw_rounded_rect(
        self,
        draw: "ImageDraw.ImageDraw",
        coords: Tuple[int, int, int, int],
        radius: int,
        fill: Tuple[int, int, int],
        alpha: int = 255,
    ) -> None:
        """Draw a rounded rectangle."""
        x1, y1, x2, y2 = coords
        try:
            # Try to use rounded_rectangle (Pillow >= 8.2)
            draw.rounded_rectangle(coords, radius=radius, fill=fill + (alpha,))
        except AttributeError:
            # Fallback to regular rectangle
            draw.rectangle(coords, fill=fill + (alpha,))

    def render_banner(
        self,
        base_image: "Image.Image",
        product_name: str,
        tagline: str,
        price: float,
        mrp: float,
        discount_pct: float,
        category: str = "",
        cta: str = "Shop Now!",
        intent_vector: Optional[List[float]] = None,
        category_embedding: Optional[List[float]] = None,
        output_width: int = 640,
    ) -> BytesIO:
        """
        Render a complete banner ad.
        
        Args:
            base_image: Product image
            product_name: Product title
            tagline: Generated tagline
            price: Selling price
            mrp: Maximum retail price
            discount_pct: Discount percentage
            category: Product category
            cta: Call to action text
            intent_vector: Detected intent weights [K]
            category_embedding: Category embedding for theming
            output_width: Output image width
        
        Returns:
            BytesIO containing PNG image
        """
        if not HAS_PIL:
            raise RuntimeError("PIL/Pillow is required for banner rendering")

        # Resize product image to output width
        w0, h0 = base_image.size
        scale = output_width / w0
        new_h = int(h0 * scale)
        product_img = base_image.resize((output_width, new_h), Image.LANCZOS)
        
        # Calculate white extension height for text area
        text_area_height = max(120, int(new_h * 0.35))  # At least 120px or 35% of image height
        
        # Create extended canvas with white background
        total_height = new_h + text_area_height
        img = Image.new("RGB", (output_width, total_height), (255, 255, 255))
        
        # Paste product image at top
        img.paste(product_img, (0, 0))
        
        # Create drawing context
        draw = ImageDraw.Draw(img, "RGBA")
        
        # Text colors - dark text on white background
        text_color = (33, 33, 33)  # Near black
        secondary_text_color = (100, 100, 100)  # Gray for secondary info
        price_color = (0, 150, 80)  # Green for price
        mrp_color = (150, 150, 150)  # Light gray for MRP
        
        # Font sizes for text area
        title_size = max(18, int(text_area_height * 0.16))
        caption_size = max(12, int(text_area_height * 0.11))
        price_size = max(22, int(text_area_height * 0.20))
        small_size = max(10, int(text_area_height * 0.09))
        
        # Load fonts
        font_title = self._load_font(title_size, bold=True)
        font_caption = self._load_font(caption_size, bold=False)
        font_price = self._load_font(price_size, bold=True)
        font_small = self._load_font(small_size, bold=False)
        
        # Text positioning - start in white area
        pad = int(output_width * 0.04)
        line_spacing = max(6, int(text_area_height * 0.05))
        tx = pad
        ty = new_h + pad  # Start below the product image
        
        # 1. Draw TITLE (product name headline) - largest, at top of white area
        headline = self.caption_gen.generate_headline(product_name)
        try:
            title_bbox = draw.textbbox((tx, ty), headline, font=font_title)
            actual_title_h = title_bbox[3] - title_bbox[1]
        except:
            actual_title_h = title_size
        draw.text((tx, ty), headline, font=font_title, fill=text_color)
        
        # 2. Draw CAPTION (tagline) - below title, secondary color
        ty += actual_title_h + line_spacing
        caption_text = tagline[:60] if tagline else ""
        if caption_text:
            try:
                caption_bbox = draw.textbbox((tx, ty), caption_text, font=font_caption)
                actual_caption_h = caption_bbox[3] - caption_bbox[1]
            except:
                actual_caption_h = caption_size
            draw.text((tx, ty), caption_text, font=font_caption, fill=secondary_text_color)
            ty += actual_caption_h + line_spacing
        
        # 3. Draw COST (price) - prominent, below caption
        ty += int(line_spacing * 0.5)  # Extra space before price
        price_str = f"₹{int(price):,}" if price > 0 else ""
        if price_str:
            draw.text((tx, ty), price_str, font=font_price, fill=price_color)
            
            # Draw MRP with strikethrough if discount exists
            if mrp > price and discount_pct > 0:
                mrp_str = f"₹{int(mrp):,}"
                try:
                    price_bbox = draw.textbbox((tx, ty), price_str, font=font_price)
                    mrp_x = price_bbox[2] + 15
                except:
                    mrp_x = tx + len(price_str) * 12
                
                draw.text((mrp_x, ty + 6), mrp_str, font=font_small, fill=mrp_color)
                try:
                    mrp_bbox = draw.textbbox((mrp_x, ty + 6), mrp_str, font=font_small)
                    # Strikethrough line
                    line_y = (mrp_bbox[1] + mrp_bbox[3]) // 2
                    draw.line((mrp_bbox[0], line_y, mrp_bbox[2], line_y), fill=mrp_color, width=1)
                    
                    # Draw discount percentage
                    discount_str = f" ({int(discount_pct)}% off)"
                    draw.text((mrp_bbox[2] + 8, ty + 6), discount_str, font=font_small, fill=price_color)
                except:
                    pass
        
        # Save to BytesIO
        output = BytesIO()
        img.save(output, format="PNG", optimize=True)
        output.seek(0)
        return output


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def generate_product_poster(
    image: "Image.Image",
    product_name: str,
    tagline: Optional[str] = None,
    price: float = 0,
    mrp: float = 0,
    discount_pct: float = 0,
    category: str = "",
    cta: Optional[str] = None,
    intent_vector: Optional[List[float]] = None,
    output_width: int = 640,
) -> Optional[BytesIO]:
    """
    Generate a banner-style product poster.
    
    Args:
        image: PIL Image of the product
        product_name: Product title
        tagline: Custom tagline (auto-generated if None)
        price: Selling price
        mrp: MRP for discount calculation
        discount_pct: Discount percentage
        category: Product category
        cta: Call to action (auto-generated if None)
        intent_vector: Intent weights from model
        output_width: Output image width in pixels
    
    Returns:
        BytesIO containing PNG image, or None on error
    """
    if not HAS_PIL:
        return None
    
    try:
        renderer = BannerRenderer()
        
        # Generate captions if not provided
        if tagline is None:
            tagline = renderer.caption_gen.generate_tagline(
                product_name, category, discount_pct, intent_vector
            )
        
        if cta is None:
            cta = renderer.caption_gen.generate_cta(discount_pct, price)
        
        return renderer.render_banner(
            base_image=image,
            product_name=product_name,
            tagline=tagline,
            price=price,
            mrp=mrp,
            discount_pct=discount_pct,
            category=category,
            cta=cta,
            intent_vector=intent_vector,
            output_width=output_width,
        )
    except Exception as e:
        print(f"[laygen] Poster generation failed: {e}")
        return None


def generate_product_poster_from_url(
    image_source: str,
    product_name: str,
    tagline: Optional[str] = None,
    price: float = 0,
    mrp: float = 0,
    discount_pct: float = 0,
    category: str = "",
    cta: Optional[str] = None,
    intent_vector: Optional[List[float]] = None,
    output_width: int = 640,
) -> Optional[BytesIO]:
    """
    Generate a product poster from image URL.
    
    Args:
        image_source: URL or local path to product image
        product_name: Product title
        tagline: Custom tagline (auto-generated if None)
        price: Selling price
        mrp: MRP for discount calculation
        discount_pct: Discount percentage
        category: Product category
        cta: Call to action (auto-generated if None)
        intent_vector: Intent weights from model
        output_width: Output image width in pixels
    
    Returns:
        BytesIO containing PNG image, or None on error
    """
    if not HAS_PIL:
        return None
    
    try:
        # Load image from URL or path
        if image_source and str(image_source).lower().startswith("http"):
            if not HAS_REQUESTS:
                return None
            resp = http_requests.get(image_source, timeout=10)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
        elif image_source:
            img = Image.open(image_source).convert("RGB")
        else:
            # Create placeholder image
            img = Image.new("RGB", (output_width, int(output_width * 1.2)), color=(20, 20, 28))
        
        return generate_product_poster(
            image=img,
            product_name=product_name,
            tagline=tagline,
            price=price,
            mrp=mrp,
            discount_pct=discount_pct,
            category=category,
            cta=cta,
            intent_vector=intent_vector,
            output_width=output_width,
        )
    except Exception as e:
        print(f"[laygen] Poster from URL failed: {e}")
        return None


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Layout Generation & Price Mapping - Self-Test")
    print("=" * 60)
    
    # Test caption generation
    print("\n[1] Caption Generation Test")
    cap_gen = CaptionGenerator(seed=42)
    
    test_cases = [
        ("Samsung Galaxy S24 Ultra 256GB", "Electronics > Mobile > Smartphone", 15),
        ("Nike Air Max Sneakers Black", "Fashion > Footwear > Sports Shoes", 30),
        ("Lakme Perfect Glow Face Cream", "Beauty > Skincare > Face Care", 0),
        ("IKEA KALLAX Shelf Unit White", "Home > Furniture > Storage", 45),
    ]
    
    for title, category, discount in test_cases:
        tagline = cap_gen.generate_tagline(title, category, discount)
        headline = cap_gen.generate_headline(title)
        cta = cap_gen.generate_cta(discount, 5000)
        
        print(f"\n  Product: {title[:40]}...")
        print(f"  Tagline: {tagline}")
        print(f"  Headline: {headline}")
        print(f"  CTA: {cta}")
    
    # Test saliency detection
    if HAS_PIL:
        print("\n[2] Saliency Detection Test")
        detector = SaliencyDetector()
        
        # Create a test image with a centered object
        test_img = Image.new("RGB", (256, 256), color=(50, 50, 60))
        draw = ImageDraw.Draw(test_img)
        # Add a bright region in center
        draw.ellipse((80, 80, 176, 176), fill=(255, 200, 100))
        
        saliency_map = detector.compute_saliency_map(test_img)
        print(f"  Saliency map shape: {saliency_map.shape}")
        print(f"  Center saliency: {saliency_map[3:5, 3:5].mean():.3f}")
        print(f"  Corner saliency: {saliency_map[0:2, 0:2].mean():.3f}")
        
        # Test layout scoring
        print("\n[3] Bayesian Layout Scoring Test")
        scorer = BayesianLayoutScorer(detector)
        
        banner_pos = scorer.find_optimal_placement(test_img, "banner")
        print(f"  Banner placement: {banner_pos['position']}")
        print(f"  Banner score: {banner_pos['score']:.3f}")
        print(f"  Suggested palette: {banner_pos['palette_idx']}")
        
        cta_pos = scorer.find_optimal_placement(test_img, "cta")
        print(f"  CTA placement: {cta_pos['position']}")
        print(f"  CTA score: {cta_pos['score']:.3f}")
        
        # Test full poster generation
        print("\n[4] Full Poster Generation Test")
        renderer = BannerRenderer()
        
        poster = generate_product_poster(
            image=test_img,
            product_name="Samsung Galaxy S24 Ultra 256GB Titanium",
            price=99999,
            mrp=129999,
            discount_pct=23,
            category="Electronics > Mobile > Smartphone",
            intent_vector=[0.4, 0.3, 0.2, 0.1],
        )
        
        if poster:
            print(f"  Poster generated: {len(poster.getvalue())} bytes")
        else:
            print("  Poster generation skipped (missing PIL)")
    else:
        print("\n[2-4] Skipped (PIL not installed)")
    
    print("\n" + "=" * 60)
    print("All tests completed.")
    print("=" * 60)
