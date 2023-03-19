import torch
from functools import partial
from easydict import EasyDict as edict
from albumentations import *

from app.SimpleClick.isegm.data.datasets import *
from app.SimpleClick.isegm.model.losses import *
from app.SimpleClick.isegm.data.transforms import *
from app.SimpleClick.isegm.engine.trainer import ISTrainer
from app.SimpleClick.isegm.model.metrics import AdaptiveIoU
from app.SimpleClick.isegm.data.points_sampler import MultiPointSampler
from app.SimpleClick.isegm.utils.log import logger
from app.SimpleClick.isegm.model import initializer

from app.SimpleClick.isegm.model.is_hrnet_model import HRNetModel
from app.SimpleClick.isegm.model.is_deeplab_model import DeeplabModel
from app.SimpleClick.isegm.model.is_segformer_model import SegformerModel
from app.SimpleClick.isegm.model.is_hrformer_model import HRFormerModel
from app.SimpleClick.isegm.model.is_swinformer_model import SwinformerModel
from app.SimpleClick.isegm.model.is_plainvit_model import PlainVitModel