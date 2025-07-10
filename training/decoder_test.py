from sam2.modeling.sam2_base import SAM2Base
from omegaconf import OmegaConf
from hydra.utils import instantiate

cfg = OmegaConf.create(yaml_config)

# Instantiate the model
model = instantiate(cfg.model)