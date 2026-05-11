import numpy as np
from utils.augment_multisize import D4_TRANSFORMS, D4_NAMES, apply_d4, seeded_augment_ids

def test_d4_transforms_preserves_mask_values():
    mask = np.zeros((3, 64, 64), dtype=np.uint8)
    mask[0, 10, 10] = 1
    mask[1, 20, 30] = 1
    for aug_id, (name, fn) in enumerate(D4_TRANSFORMS):
        result = fn(mask.copy(), axes=(-2, -1))
        assert result.shape == mask.shape
        assert result.dtype == np.uint8
        assert int(result.sum()) == int(mask.sum())
        assert set(np.unique(result)).issubset({0, 1})

def test_d4_names_length_matches():
    assert len(D4_NAMES) == 8
    assert D4_NAMES[0] == "identity"
    assert D4_NAMES[1] == "rot90"

def test_apply_d4_calls_correct_transform():
    image = np.random.rand(64, 64).astype(np.float32)
    mask = np.zeros((2, 64, 64), dtype=np.uint8)
    result_img, result_mask = apply_d4(image, mask, aug_id=2, axes=(-2,-1))
    expected_img = np.rot90(image, k=2, axes=(-2,-1))
    np.testing.assert_array_equal(result_img, expected_img)

def test_apply_d4_mask_shape_preserved():
    image = np.random.rand(64, 64).astype(np.float32)
    mask = np.random.randint(0, 2, (3, 64, 64)).astype(np.uint8)
    for aug_id in range(8):
        img_out, mask_out = apply_d4(image, mask, aug_id, axes=(-2,-1))
        assert mask_out.shape == mask.shape

def test_seeded_augment_ids_reproducible():
    ids1 = list(seeded_augment_ids("tile_001", n=4, global_seed=42))
    ids2 = list(seeded_augment_ids("tile_001", n=4, global_seed=42))
    assert ids1 == ids2

def test_seeded_augment_ids_different_seeds():
    ids1 = list(seeded_augment_ids("tile_001", n=4, global_seed=42))
    ids2 = list(seeded_augment_ids("tile_001", n=4, global_seed=99))
    assert ids1 != ids2

def test_seeded_augment_ids_no_duplicates_when_n_le_7():
    ids = list(seeded_augment_ids("tile_x", n=7, global_seed=42))
    assert len(ids) == 7
    assert len(set(ids)) == 7  # no duplicates
    assert 0 not in ids  # identity excluded

def test_seeded_augment_ids_length():
    ids = list(seeded_augment_ids("tile_x", n=4, global_seed=42))
    assert len(ids) == 4
    assert all(0 <= i < 8 for i in ids)
