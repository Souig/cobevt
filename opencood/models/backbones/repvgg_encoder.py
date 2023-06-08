import torch
import torch.nn as nn
import torchvision.models as models
from RepVGG.repvgg import get_RepVGG_func_by_name

from einops import rearrange


class RepVGGEncoder(nn.Module):
    """
    RepVGG family to encode image.

    Parameters
    ----------
    params: dict
        The parameters of RepVGG.
    """

    def __init__(self, params):
        super(RepVGGEncoder, self).__init__()

        self.backbone_name = params['backbone_name']
        self.pretrained = params['pretrained']
        image_height = params['image_height']
        image_width = params['image_width']
        deploy = params['deploy']
        backbone_file = params['backbone_file']
        self.idx_pick = params['id_pick']

        # resnets = {18: models.resnet18,
        #            34: models.resnet34,
        #            50: models.resnet50,
        #            101: models.resnet101,
        #            152: models.resnet152}
        backbone_names = ['RepVGG-A0', 'RepVGG-A1', 'RepVGG-A2', 'RepVGG-B0', 'RepVGG-B1', 'RepVGG-B1g2', 'RepVGG-B1g4',
                          'RepVGG-B2', 'RepVGG-B2g2', 'RepVGG-B2g4', 'RepVGG-B3', 'RepVGG-B3g2', 'RepVGG-B3g4','RepVGG-A1-slim']

        if self.backbone_name not in backbone_names:
            raise ValueError(
                "{} is not a valid backbone of RepVGG "
                "layers".format(self.backbone_name))

        backbone = get_RepVGG_func_by_name(self.backbone_name)
        self.encoder = backbone(deploy)
        if self.pretrained:
            checkpoint = torch.load(backbone_file)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            ckpt = {k.replace('module.', ''): v for k, v in checkpoint.items()}  # strip the names
            self.encoder.load_state_dict(ckpt)

        # Pass a dummy tensor to precompute intermediate shapes
        dummy = torch.rand(1, 1, 1, image_height, image_width, 3)
        output_shapes = [x.shape for x in self(dummy)]

        self.output_shapes = output_shapes

    def forward(self, input_images):
        """
        Compute deep features from input images.
        todo: multi-scale feature support

        Parameters
        ----------
        input_images : torch.Tensor
            The input images have shape of (B,L,M,H,W,3), where L, M are
            the num of agents and num of cameras per agents.

        Returns
        -------
        features: torch.Tensor
            The deep features for each image with a shape of (B,L,M,C,H,W)
        """
        b, l, m, h, w, c = input_images.shape
        input_images = input_images.view(b * l * m, h, w, c)
        # b, h, w, c -> b, c, h, w
        input_images = input_images.permute(0, 3, 1, 2).contiguous()

        x = self.encoder.stage0(input_images)

        x0 = x
        for module in self.encoder.stage1:
            x0 = module(x0)
        x1 = x0
        for module in self.encoder.stage2:
            x1 = module(x1)
        x2 = x1
        for module in self.encoder.stage3:
            x2 = module(x2)
        x3 = x2
        for module in self.encoder.stage4:
            x3 = module(x3)

        x0 = rearrange(x0, '(b l m) c h w -> b l m c h w',
                       b=b, l=l, m=m)
        x1 = rearrange(x1, '(b l m) c h w -> b l m c h w',
                       b=b, l=l, m=m)
        x2 = rearrange(x2, '(b l m) c h w -> b l m c h w',
                       b=b, l=l, m=m)
        x3 = rearrange(x3, '(b l m) c h w -> b l m c h w',
                       b=b, l=l, m=m)
        results = [x0, x1, x2, x3]

        if isinstance(self.idx_pick, list):
            return [results[i] for i in self.idx_pick]
        else:
            return results[self.idx_pick]


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from torch.utils.data import DataLoader
    from opencood.tools import train_utils

    params = load_yaml('/home/ubuntu/work/SHU_Project/cobevt/opencood/hypes_yaml/opcamera/corpbevt.yaml')

    opencood_train_dataset = build_dataset(params, visualize=False, train=True)
    data_loader = DataLoader(opencood_train_dataset,
                             batch_size=4,
                             num_workers=8,
                             collate_fn=opencood_train_dataset.collate_batch,
                             shuffle=False,
                             pin_memory=False)
    
    repvgg_params = {
        'backbone_name': 'RepVGG-A1',
        'pretrained': True,
        'image_width': 224,
        'image_height': 224,
        'deploy': False,
        'backbone_file': '/home/ubuntu/work/SHU_Project/cobevt/RepVGG/pretrained/RepVGG-A1-train.pth',
        'id_pick': [1, 3]}

    model = RepVGGEncoder(repvgg_params)
    model.cuda()
    device = torch.device('cuda')

    for j, batch_data in enumerate(data_loader):
        cam_data = train_utils.to_device(batch_data['ego']['inputs'],
                                         device)
        output = model(cam_data)
        print('test passed')
