import copy
import importlib

import torch
import torch.nn as nn
from collections.abc import Mapping
import cv2
import numpy as np

class ModelHelper(nn.Module):
    """Build model from cfg"""

    def __init__(self, cfg):
        super(ModelHelper, self).__init__()

        self.frozen_layers = []
        for cfg_subnet in cfg[0:2]:     # 取backbone和neck
            mname = cfg_subnet["name"]
            kwargs = cfg_subnet["kwargs"]
            mtype = cfg_subnet["type"]
            if cfg_subnet.get("frozen", False):
                self.frozen_layers.append(mname)
            if cfg_subnet.get("prev", None) is not None:
                prev_module = getattr(self, cfg_subnet["prev"])
                kwargs["inplanes"] = prev_module.get_outplanes()
                kwargs["instrides"] = prev_module.get_outstrides()

            module = self.build(mtype, kwargs)
            self.add_module(mname, module)
            # break       # PAD<-比UniAD就改了这

    def build(self, mtype, kwargs):
        module_name, cls_name = mtype.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, cls_name)
        return cls(**kwargs)

    def cuda(self):
        self.device = torch.device("cuda")
        return super(ModelHelper, self).cuda()

    def cpu(self):
        self.device = torch.device("cpu")
        return super(ModelHelper, self).cpu()

    def forward(self, input):
        # import pdb; pdb.set_trace()
        input = copy.copy(input)
        if input.device != self.device:
            # input = to_device(input, device=self.device)
            input=input.cuda()
        for submodule in self.children():
            output = submodule(input)
            input = output
            # input.update(output)
        feat=[]
        size=(224,224)
        return output["feature_align"]

    def freeze_layer(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        """
        Sets the module in training mode.
        This has any effect only on modules such as Dropout or BatchNorm.

        Returns:
            Module: self
        """
        self.training = mode
        for mname, module in self.named_children():
            if mname in self.frozen_layers:
                self.freeze_layer(module)
            else:
                module.train(mode)
        return self
