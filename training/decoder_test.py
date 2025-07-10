from sam2.modeling.sam2_base import SAM2Base
from omegaconf import OmegaConf
from hydra.utils import instantiate

config = OmegaConf.load("/scratch_net/ken/radjoe/Projects/SourceCode/MedSAM2/sam2/configs/test_config.yaml")

model = instantiate(config.trainer.model)