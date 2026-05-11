from yacs.config import CfgNode as CN

cfg = CN()

# GLOBAL
cfg.global_seed = 42

# MODEL
cfg.model = CN()
cfg.model.encoder = "default"
cfg.model.pretrained = False
cfg.model.skip_ch = 64
cfg.model.use_cgm = False
cfg.model.aux_losses = 0
cfg.model.dropout = 0.3
cfg.model.name = "Unet3"

# DATA
cfg.data = CN()
cfg.data.train_root = "datasets/processed/20260512_tiles_64"
cfg.data.test_root = ""
cfg.data.num_classes = 2
cfg.data.class_names = ["1.0", "2.8"]
cfg.data.input_channels = 1
cfg.data.tile_size = 64
cfg.data.stride = 32
cfg.data.batch_size = 32
cfg.data.num_workers = 4

# HYPERPARAMETERS
cfg.train = CN()
cfg.train.seed = 42
cfg.train.epochs = 50
cfg.train.lr = 0.001
cfg.train.lrf = 0.0005  # final lr
cfg.train.scheduler = "cyclic"
cfg.train.warmup_iters = 500
cfg.train.optimizer = "adamw"
cfg.train.weight_decay = 0.0001
cfg.train.momentum = 0.9
cfg.train.nesterov = True
cfg.train.accum_steps = 1
cfg.train.resume = ""
cfg.train.val_interval = 1
cfg.train.device = "cuda"
cfg.train.aux_weight = 0.4
cfg.train.loss_type = "focal_dice"
cfg.train.save_name = "multisize_kclass"
cfg.train.log_dir = "./runs/multisize"
cfg.train.class_weights = None
cfg.train.focal_gamma = 2.0
cfg.train.dice_weight = 1.0
cfg.train.focal_weight = 1.0
cfg.train.count_loss_weight = 0.0

# POSTPROCESS
cfg.postprocess = CN()
cfg.postprocess.thresholds = [0.5, 0.5]
cfg.postprocess.min_distances = [3, 5]
cfg.postprocess.match_radius = 3

# TARGET
cfg.target = CN()
cfg.target.mode = "raw"  # "raw" or "disk"
cfg.target.radius_per_class = [1, 3]
cfg.target.keep_raw_center_mask = True