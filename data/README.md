# PixelMatch — synthetic data generators

This directory holds the offline data-generation scripts.  The artefacts
they produce are deliberately ignored from git (see `.gitignore`).

## Files

| Script | Output | Description |
| --- | --- | --- |
| `generate_catalog.py` | `catalog.csv` + `images/{pid}.png` | 100K product rows with title, description, category, brand, price, color, and a procedurally generated image. |
| `generate_interactions.py` | `interactions.csv` + `cold_product_ids.npy` | 1M `(user, product, type, rating, timestamp)` rows drawn from a Zipfian popularity distribution with ~10K cold products. |

## Quick start

```bash
# Full scale (used for benchmarks)
python data/generate_catalog.py --num 100000 --images 2000
python data/generate_interactions.py --num 1000000

# Tiny scale (used by unit tests and laptop notebooks)
python data/generate_catalog.py --num 2000 --images 200
python data/generate_interactions.py --num 20000 --users 1000 --products 2000
```

Both scripts are seeded (`--seed 42` by default) so runs are bit-for-bit
reproducible.

## Schema notes

* `catalog.csv` `has_image` indicates whether a PNG was rendered for that
  row.  Images are *optional* — the rest of the pipeline can run with
  zero PNGs because the image encoder will produce a deterministic
  fallback vector.
* `interactions.csv` `is_cold_product` is a convenience flag mirroring
  the `cold_product_ids.npy` artifact and is consumed by
  `pixelmatch.evaluation.cold_start`.

## Storage footprint

At full scale the catalog CSV is ~80 MB and the (subset) image directory
is ~150 MB.  Interactions at 1M rows is ~40 MB.
