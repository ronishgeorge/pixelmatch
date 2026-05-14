"""Generate a synthetic product catalog for PixelMatch.

Produces (by default) 100,000 products with realistic-ish titles,
descriptions, and procedurally generated 224x224 images.  Output:

    data/catalog.csv             — metadata table
    data/images/{product_id}.png — image per product (optional, sampled)

Usage
-----
    python data/generate_catalog.py --num 100000 --images 5000

The ``--images`` flag controls how many products get a saved PNG (writing
all 100K images is expensive — for the headline benchmark we encode images
in-memory and only persist a thumbnail subset for inspection).
"""

from __future__ import annotations

import argparse
import logging
import os
import random
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

CATEGORIES = [
    "apparel", "footwear", "electronics", "home_decor", "kitchen",
    "beauty", "books", "sports", "toys", "outdoor", "office", "pet_supplies",
]

BRANDS = [
    "Nimbus", "Vertex", "Lumen", "Halcyon", "Atlas", "Stratus", "Cobalt",
    "Ember", "Solace", "Onyx", "Veridian", "Crescent", "Pinnacle", "Marlow",
    "Quill", "Spectra", "Talon", "Wren", "Zephyr", "Beacon",
]

COLORS = [
    "midnight black", "ivory white", "sage green", "navy blue", "burnt orange",
    "blush pink", "charcoal grey", "olive", "burgundy", "teal", "mustard yellow",
    "lavender", "cream", "rust", "forest green", "slate", "denim", "rose gold",
]

CATEGORY_DESCRIPTORS = {
    "apparel": ["soft", "breathable", "lightweight", "tailored", "relaxed-fit", "wrinkle-resistant"],
    "footwear": ["cushioned", "non-slip", "waterproof", "lightweight", "memory-foam", "shock-absorbing"],
    "electronics": ["wireless", "low-latency", "energy-efficient", "fast-charging", "noise-cancelling"],
    "home_decor": ["handcrafted", "minimalist", "rustic", "modern", "artisan", "vintage"],
    "kitchen": ["non-stick", "dishwasher-safe", "BPA-free", "stainless steel", "induction-ready"],
    "beauty": ["paraben-free", "vegan", "dermatologist-tested", "fragrance-free", "long-lasting"],
    "books": ["paperback", "bestselling", "illustrated", "hardcover", "annotated", "unabridged"],
    "sports": ["high-performance", "moisture-wicking", "durable", "ergonomic", "competition-grade"],
    "toys": ["BPA-free", "STEM", "educational", "battery-powered", "soft-touch", "award-winning"],
    "outdoor": ["weather-resistant", "UV-protective", "ultralight", "ripstop", "windproof"],
    "office": ["ergonomic", "adjustable", "noise-dampening", "easy-glide", "modular"],
    "pet_supplies": ["chew-resistant", "vet-recommended", "machine-washable", "non-toxic"],
}


# ---------------------------------------------------------------------- #
def _title(rng: random.Random, category: str, brand: str, color: str) -> str:
    base_words = {
        "apparel": ["Crewneck Tee", "Linen Shirt", "Hooded Pullover", "Tailored Trousers", "Quilted Jacket"],
        "footwear": ["Trail Runner", "Leather Sneaker", "Suede Loafer", "Hiking Boot", "Performance Slide"],
        "electronics": ["Wireless Earbuds", "Bluetooth Speaker", "Mechanical Keyboard", "USB-C Hub", "Smart Bulb"],
        "home_decor": ["Ceramic Vase", "Linen Throw", "Brass Lantern", "Woven Basket", "Wall Mirror"],
        "kitchen": ["Cast-Iron Skillet", "Pour-Over Kettle", "Bamboo Cutting Board", "French Press", "Chef Knife"],
        "beauty": ["Vitamin C Serum", "Hydrating Mist", "Matte Lipstick", "Argan Hair Oil", "Clay Mask"],
        "books": ["Novel", "Cookbook", "Travel Guide", "Memoir", "Field Guide"],
        "sports": ["Yoga Mat", "Resistance Band Set", "Running Belt", "Compression Sleeve", "Cycling Bottle"],
        "toys": ["Building Block Set", "Plush Animal", "Wooden Puzzle", "Remote-Control Car", "Activity Book"],
        "outdoor": ["Camping Hammock", "Down Jacket", "Trekking Pole", "Headlamp", "Tarp Shelter"],
        "office": ["Ergonomic Chair", "Standing Desk Mat", "Document Organizer", "Task Lamp", "Notebook Set"],
        "pet_supplies": ["Chew Toy", "Travel Carrier", "Grooming Brush", "Treat Pouch", "Orthopedic Bed"],
    }
    word = rng.choice(base_words[category])
    return f"{brand} {color.title()} {word}"


def _description(rng: random.Random, category: str, words: int) -> str:
    descriptors = CATEGORY_DESCRIPTORS[category]
    benefits = [
        "Designed for everyday use.",
        "Perfect for gifting.",
        "Crafted with sustainable materials.",
        "Backed by a one-year warranty.",
        "Engineered for comfort and durability.",
        "Hand-finished by skilled artisans.",
        "Independently lab-tested for quality.",
        "Optimized for low-light conditions.",
        "Built to last through daily wear.",
        "Reduces waste with refillable design.",
    ]
    sentences = [
        f"This {rng.choice(descriptors)} {category.replace('_', ' ')} item delivers reliable performance.",
        rng.choice(benefits),
        f"Features include {rng.choice(descriptors)} construction and {rng.choice(descriptors)} finish.",
        f"Suitable for {rng.choice(['indoor', 'outdoor', 'travel', 'studio', 'everyday'])} use.",
        rng.choice(benefits),
        f"Customers love its {rng.choice(descriptors)} feel and {rng.choice(descriptors)} look.",
    ]
    text = " ".join(rng.choices(sentences, k=max(3, words // 12)))
    return text


# ---------------------------------------------------------------------- #
def _generate_image(
    out_path: str | None,
    category: str,
    color_idx: int,
    size: int = 224,
    seed: int = 0,
) -> Image.Image:
    """Procedurally generate a 224x224 PNG: gradient background + category shape."""
    rng = random.Random(seed)
    img = Image.new("RGB", (size, size), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Gradient background
    c1 = (rng.randint(40, 220), rng.randint(40, 220), rng.randint(40, 220))
    c2 = (rng.randint(40, 220), rng.randint(40, 220), rng.randint(40, 220))
    for y in range(size):
        t = y / size
        r = int(c1[0] * (1 - t) + c2[0] * t)
        g = int(c1[1] * (1 - t) + c2[1] * t)
        b = int(c1[2] * (1 - t) + c2[2] * t)
        draw.line([(0, y), (size, y)], fill=(r, g, b))

    # Category-specific shape (deterministic per category for clustering)
    cat_idx = CATEGORIES.index(category) if category in CATEGORIES else 0
    fg = ((color_idx * 37) % 255, (cat_idx * 53) % 255, (color_idx * 19 + cat_idx * 11) % 255)
    cx, cy = size // 2, size // 2
    if cat_idx % 4 == 0:  # circle
        draw.ellipse([cx - 60, cy - 60, cx + 60, cy + 60], fill=fg)
    elif cat_idx % 4 == 1:  # square
        draw.rectangle([cx - 50, cy - 50, cx + 50, cy + 50], fill=fg)
    elif cat_idx % 4 == 2:  # triangle
        draw.polygon([(cx, cy - 60), (cx - 60, cy + 50), (cx + 60, cy + 50)], fill=fg)
    else:  # bars
        for i in range(-2, 3):
            draw.rectangle([cx - 50 + i * 25, cy - 50, cx - 35 + i * 25, cy + 50], fill=fg)

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        img.save(out_path, optimize=True)
    return img


# ---------------------------------------------------------------------- #
def generate_catalog(
    num_products: int = 100_000,
    output_dir: str = "data",
    image_subset: int = 2_000,
    seed: int = 42,
    categories: Sequence[str] = CATEGORIES,
) -> pd.DataFrame:
    """Build the synthetic catalog and persist ``catalog.csv``."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)
    img_dir = os.path.join(output_dir, "images")

    rows = []
    image_indices = set(np_rng.choice(num_products, size=min(image_subset, num_products), replace=False).tolist())

    for pid in range(num_products):
        category = rng.choice(categories)
        brand = rng.choice(BRANDS)
        color_idx = rng.randint(0, len(COLORS) - 1)
        color = COLORS[color_idx]
        title = _title(rng, category, brand, color)
        desc = _description(rng, category, words=rng.randint(50, 200))
        price = round(np_rng.lognormal(mean=3.0, sigma=0.6), 2)
        rows.append(
            {
                "product_id": pid,
                "title": title,
                "description": desc,
                "category": category,
                "brand": brand,
                "color": color,
                "price": price,
                "has_image": pid in image_indices,
            }
        )

        if pid in image_indices:
            out_path = os.path.join(img_dir, f"{pid}.png")
            _generate_image(out_path, category=category, color_idx=color_idx, seed=seed + pid)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "catalog.csv")
    df.to_csv(csv_path, index=False)
    logger.info("Wrote %d products to %s (and %d images to %s)", len(df), csv_path, len(image_indices), img_dir)
    return df


# ---------------------------------------------------------------------- #
def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate PixelMatch synthetic catalog")
    p.add_argument("--num", type=int, default=100_000, help="number of products")
    p.add_argument("--images", type=int, default=2_000, help="number of products to render PNGs for")
    p.add_argument("--output-dir", type=str, default="data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(levelname)s %(message)s")
    generate_catalog(
        num_products=args.num,
        output_dir=args.output_dir,
        image_subset=args.images,
        seed=args.seed,
    )


if __name__ == "__main__":
    _cli()
