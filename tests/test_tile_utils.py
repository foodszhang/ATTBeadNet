import numpy as np
from utils.tile_utils import sliding_window, make_overlay

def test_sliding_window_yields_correct_coords():
    image = np.random.rand(100, 100).astype(np.float32)
    mask = np.zeros((2, 100, 100), dtype=np.uint8)
    tiles = list(sliding_window(image, mask, tile_size=64, stride=32))
    coords = [(t["y0"], t["x0"]) for t in tiles]
    assert (0, 0) in coords
    assert (32, 0) in coords
    assert (0, 32) in coords
    for t in tiles:
        assert t["y0"] + 64 <= 100
        assert t["x0"] + 64 <= 100

def test_sliding_window_respects_stride():
    image = np.random.rand(100, 100).astype(np.float32)
    mask = np.zeros((2, 100, 100), dtype=np.uint8)
    tiles = list(sliding_window(image, mask, tile_size=64, stride=48))
    assert len(tiles) == 1  # stride 48 means only y0=0 fits

def test_sliding_window_image_and_mask_shapes():
    image = np.random.rand(200, 150).astype(np.float32)
    mask = np.random.randint(0, 2, (3, 200, 150)).astype(np.uint8)
    for t in sliding_window(image, mask, tile_size=64, stride=32):
        assert t["image"].shape == (64, 64)
        assert t["mask"].shape == (3, 64, 64)
        assert t["image"].dtype == np.float32
        assert t["mask"].dtype == np.uint8

def test_sliding_window_count_per_class():
    mask = np.zeros((2, 100, 100), dtype=np.uint8)
    mask[0, 10, 10] = 1
    mask[0, 20, 20] = 1
    mask[1, 30, 30] = 1
    image = np.random.rand(100, 100).astype(np.float32)
    tiles = list(sliding_window(image, mask, tile_size=64, stride=32))
    # tile at y0=0,x0=0 should have count_class_0=2, count_class_1=1
    t = tiles[0]
    assert t["count_per_class"] == [2, 1]
    assert t["count_total"] == 3
    assert t["is_positive"] == True

def test_make_overlay_colors_per_class():
    image = np.random.rand(64, 64).astype(np.float32)
    mask = np.zeros((2, 64, 64), dtype=np.uint8)
    mask[0, 10, 10] = 1
    mask[1, 30, 30] = 1
    overlay = make_overlay(image, mask, class_names=["1.0", "2.8"])
    assert overlay.shape == (64, 64, 3)
    assert overlay.dtype == np.uint8
    # class 0 center should be cyan (0, 255, 255)
    assert tuple(overlay[10, 10]) == (0, 255, 255), f"expected cyan, got {overlay[10,10]}"
    # class 1 center should be yellow (255, 255, 0)
    assert tuple(overlay[30, 30]) == (255, 255, 0), f"expected yellow, got {overlay[30,30]}"

def test_make_overlay_3_classes():
    mask = np.zeros((3, 64, 64), dtype=np.uint8)
    mask[0, 10, 10] = 1
    mask[1, 20, 20] = 1
    mask[2, 30, 30] = 1
    image = np.random.rand(64, 64).astype(np.float32)
    overlay = make_overlay(image, mask, class_names=["1.0", "2.8", "4.5"])
    # 3 classes: H=0/120/240, all fully saturated
    # class 0 at hue 0 deg = (255, 0, 0) in RGB after colorsys
    # class 1 at hue 120 deg = (0, 255, 0)
    # class 2 at hue 240 deg = (0, 0, 255)
    assert tuple(overlay[10, 10]) == (255, 0, 0), f"class0 expected red, got {overlay[10,10]}"
    assert tuple(overlay[20, 20]) == (0, 255, 0), f"class1 expected green, got {overlay[20,20]}"
    assert tuple(overlay[30, 30]) == (0, 0, 255), f"class2 expected blue, got {overlay[30,30]}"

def test_sliding_window_no_padding():
    """Edge tiles that would exceed bounds are skipped."""
    image = np.random.rand(100, 100).astype(np.float32)
    mask = np.zeros((2, 100, 100), dtype=np.uint8)
    tiles = list(sliding_window(image, mask, tile_size=64, stride=1))
    # With stride 1, only positions where y0+64<=100 and x0+64<=100 are valid
    for t in tiles:
        assert t["y0"] + 64 <= 100
        assert t["x0"] + 64 <= 100
