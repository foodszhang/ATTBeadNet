import os, tempfile, numpy as np, torch, tifffile
import importlib.util
spec = importlib.util.spec_from_file_location("multisize_dataset", "/home/foods/pro/ATTBeadNet/datasets/multisize_dataset.py")
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)
MultiSizeBeadTileDataset = _mod.MultiSizeBeadTileDataset
MultiSizeTileManifest = _mod.MultiSizeTileManifest

def test_manifest_csv_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = os.path.join(tmpdir, "manifest.csv")
        content = """\
tile_id,source_dataset,source_image_id,split,image_path,mask_path,y0,x0,tile_size,stride,class_names,count_class_0,count_class_1,count_total,is_positive,augmentation_id,geom_transform,source_tile_id,is_augmented
20260511_1_y0_x0,datasets/20260511,1,train_pool,images/20260511_1_y0_x0.tif,masks/20260511_1_y0_x0_mask.npy,0,0,64,32,"1.0,2.8",3,1,4,True,0,identity,20260511_1_y0_x0,False
"""
        with open(manifest_path, "w") as f:
            f.write(content.lstrip("\n"))

        # write dummy image and mask
        os.makedirs(os.path.join(tmpdir, "images"), exist_ok=True)
        os.makedirs(os.path.join(tmpdir, "masks"), exist_ok=True)
        img = np.random.rand(1, 64, 64).astype(np.float32)
        mask = np.random.randint(0, 2, (2, 64, 64)).astype(np.uint8)
        tifffile.imwrite(os.path.join(tmpdir, "images", "20260511_1_y0_x0.tif"), img)
        np.save(os.path.join(tmpdir, "masks", "20260511_1_y0_x0_mask.npy"), mask)

        ds = MultiSizeBeadTileDataset(tmpdir, split="train_pool")
        sample = ds[0]
        assert sample["image"].shape == (1, 64, 64)
        assert sample["mask"].shape == (2, 64, 64)
        assert sample["count"].shape == (2,)
        assert sample["count"].dtype == torch.int64
        assert sample["tile_id"] == "20260511_1_y0_x0"
        assert sample["class_names"] == ["1.0", "2.8"]
        assert isinstance(sample["source_image_id"], str)

def test_dataset_filters_by_split():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "images"), exist_ok=True)
        os.makedirs(os.path.join(tmpdir, "masks"), exist_ok=True)
        manifest = """\
tile_id,source_dataset,source_image_id,split,image_path,mask_path,y0,x0,tile_size,stride,class_names,count_class_0,count_class_1,count_total,is_positive,augmentation_id,geom_transform,source_tile_id,is_augmented
tileA,d,1,internal_val,iA,mA,0,0,64,32,"1.0,2.8",0,0,0,False,0,identity,tileA,False
tileB,d,2,train_pool,iB,mB,0,0,64,32,"1.0,2.8",0,0,0,False,0,identity,tileB,False
"""
        with open(os.path.join(tmpdir, "manifest.csv"), "w") as f:
            f.write(manifest.lstrip("\n"))

        img = np.random.rand(1, 64, 64).astype(np.float32)
        mask = np.random.randint(0, 2, (2, 64, 64)).astype(np.uint8)
        tifffile.imwrite(os.path.join(tmpdir, "images", "tileA.tif"), img)
        tifffile.imwrite(os.path.join(tmpdir, "images", "tileB.tif"), img)
        np.save(os.path.join(tmpdir, "masks", "tileA_mask.npy"), mask)
        np.save(os.path.join(tmpdir, "masks", "tileB_mask.npy"), mask)

        ds_val = MultiSizeBeadTileDataset(tmpdir, split="internal_val")
        ds_train = MultiSizeBeadTileDataset(tmpdir, split="train_pool")
        assert len(ds_val) == 1
        assert len(ds_train) == 1

def test_count_per_class():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "images"), exist_ok=True)
        os.makedirs(os.path.join(tmpdir, "masks"), exist_ok=True)
        manifest = """\
tile_id,source_dataset,source_image_id,split,image_path,mask_path,y0,x0,tile_size,stride,class_names,count_class_0,count_class_1,count_total,is_positive,augmentation_id,geom_transform,source_tile_id,is_augmented
tileX,d,1,train_pool,images/tileX.tif,masks/tileX_mask.npy,0,0,64,32,"1.0,2.8",5,3,8,True,0,identity,tileX,False
"""
        with open(os.path.join(tmpdir, "manifest.csv"), "w") as f:
            f.write(manifest.lstrip("\n"))

        img = np.random.rand(1, 64, 64).astype(np.float32)
        mask = np.zeros((2, 64, 64), dtype=np.uint8)
        mask[0, 5:10, 5:10] = 1  # 25 ones
        mask[1, 3, 3] = 1
        tifffile.imwrite(os.path.join(tmpdir, "images", "tileX.tif"), img)
        np.save(os.path.join(tmpdir, "masks", "tileX_mask.npy"), mask)

        ds = MultiSizeBeadTileDataset(tmpdir, split="train_pool")
        sample = ds[0]
        # count should reflect actual mask nonzero
        assert sample["count"][0] == 25
        assert sample["count"][1] == 1