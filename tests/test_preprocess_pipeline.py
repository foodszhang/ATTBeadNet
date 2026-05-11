"""Smoke tests for preprocess_multisize_tiles.py pipeline functions."""

import os
import sys
import tempfile

import numpy as np
import tifffile

# Ensure project root is on path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.preprocess_multisize_tiles import (
    collect_image_ids,
    normalize_image,
    split_by_image_id,
)


def test_collect_image_ids():
    """Test that collect_image_ids finds images and validates masks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rf_dir = os.path.join(tmpdir, "RF")
        os.makedirs(rf_dir)
        for i in range(1, 4):
            tifffile.imwrite(
                os.path.join(rf_dir, f"{i}.tif"),
                np.random.rand(10, 10).astype(np.float32),
            )
        for cd in ["1.0_mask", "2.8_mask"]:
            mdir = os.path.join(tmpdir, cd)
            os.makedirs(mdir)
            for i in range(1, 4):
                tifffile.imwrite(
                    os.path.join(mdir, f"{i}_Mask.tif"),
                    np.zeros((10, 10), dtype=np.uint8),
                )
        ids = collect_image_ids(
            tmpdir, "RF", ["1.0_mask", "2.8_mask"],
            "_Mask.tif", ".tif", allow_missing=False,
        )
        assert set(ids) == {"1", "2", "3"}


def test_collect_image_ids_missing_mask():
    """Test that collect_image_ids raises FileNotFoundError for missing mask."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rf_dir = os.path.join(tmpdir, "RF")
        os.makedirs(rf_dir)
        for i in range(1, 4):
            tifffile.imwrite(
                os.path.join(rf_dir, f"{i}.tif"),
                np.random.rand(10, 10).astype(np.float32),
            )
        # Only create mask for one class_dir
        mdir = os.path.join(tmpdir, "1.0_mask")
        os.makedirs(mdir)
        tifffile.imwrite(
            os.path.join(mdir, "1_Mask.tif"),
            np.zeros((10, 10), dtype=np.uint8),
        )
        try:
            collect_image_ids(
                tmpdir, "RF", ["1.0_mask", "2.8_mask"],
                "_Mask.tif", ".tif", allow_missing=False,
            )
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass  # expected


def test_collect_image_ids_allow_missing():
    """Test that collect_image_ids skips images with missing masks when allow_missing=True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rf_dir = os.path.join(tmpdir, "RF")
        os.makedirs(rf_dir)
        for i in range(1, 4):
            tifffile.imwrite(
                os.path.join(rf_dir, f"{i}.tif"),
                np.random.rand(10, 10).astype(np.float32),
            )
        # Create masks for 1.0_mask for all images, but only for 2.8_mask for image "1"
        mdir_10 = os.path.join(tmpdir, "1.0_mask")
        os.makedirs(mdir_10)
        for i in range(1, 4):
            tifffile.imwrite(
                os.path.join(mdir_10, f"{i}_Mask.tif"),
                np.zeros((10, 10), dtype=np.uint8),
            )
        # Only image "1" has a mask in 2.8_mask
        mdir_28 = os.path.join(tmpdir, "2.8_mask")
        os.makedirs(mdir_28)
        tifffile.imwrite(
            os.path.join(mdir_28, "1_Mask.tif"),
            np.zeros((10, 10), dtype=np.uint8),
        )
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ids = collect_image_ids(
                tmpdir, "RF", ["1.0_mask", "2.8_mask"],
                "_Mask.tif", ".tif", allow_missing=True,
            )
            assert len(w) >= 1
        # Only image 1 has all masks
        assert ids == ["1"]


def test_normalize_image_percentile():
    """Test that normalize_image clips to p1-p99.8 and rescales to [0,1]."""
    rng = np.random.default_rng(42)
    img = rng.normal(500, 100, size=(100, 100)).astype(np.float32)
    norm = normalize_image(img, p_low=1, p_high=99.8)
    assert norm.min() >= 0.0, f"min={norm.min()}"
    assert norm.max() <= 1.0, f"max={norm.max()}"
    assert norm.dtype == np.float32


def test_normalize_image_constant():
    """Test that normalize_image handles constant images gracefully."""
    img = np.full((10, 10), 500.0, dtype=np.float32)
    norm = normalize_image(img)
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0


def test_split_by_image_id_all_train():
    """Test all_train mode returns all IDs in train."""
    ids = ["1", "2", "3", "4", "5"]
    train_ids, val_ids = split_by_image_id(
        ids, split_mode="all_train", val_ratio=0.2, global_seed=42,
    )
    assert set(train_ids) == set(ids)
    assert val_ids == []


def test_split_by_image_id_train_val_by_image():
    """Test train_val_by_image splits correctly."""
    ids = ["1", "2", "3", "4", "5"]
    train_ids, val_ids = split_by_image_id(
        ids, split_mode="train_val_by_image", val_ratio=0.2, global_seed=42,
    )
    assert len(train_ids) + len(val_ids) == 5
    assert len(val_ids) == 1  # 20% of 5 = 1
    assert set(train_ids + val_ids) == set(ids)


def test_split_by_image_id_fixed_test():
    """Test fixed_test mode returns all IDs in val."""
    ids = ["1", "2", "3", "4", "5"]
    train_ids, val_ids = split_by_image_id(
        ids, split_mode="fixed_test", val_ratio=0.2, global_seed=42,
    )
    assert train_ids == []
    assert set(val_ids) == set(ids)


def test_split_by_image_id_reproducible():
    """Test that split_by_image_id is reproducible with same seed."""
    ids = ["1", "2", "3", "4", "5"]
    train_a, val_a = split_by_image_id(ids, "train_val_by_image", 0.2, 42)
    train_b, val_b = split_by_image_id(ids, "train_val_by_image", 0.2, 42)
    assert train_a == train_b
    assert val_a == val_b
