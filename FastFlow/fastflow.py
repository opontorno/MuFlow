import FrEIA.framework as Ff
import FrEIA.modules as Fm
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import constants as const

import numpy as np
import pdb


def gaussian_nll_loss(output, mu, cov, log_jac_det):
    
    B, d = output.shape
    
    cov_inv = torch.linalg.inv(cov)

    diff = (output - mu).reshape(B, d, 1) # Shape: (B, d, 1)  TODO: controllare shape output

    mahalanobis = torch.matmul(diff.transpose(1, 2), torch.matmul(cov_inv, diff)).squeeze() # Mahalanobis distance: (x - mu)^T Σ^{-1} (x - mu)
    #log_det_cov = torch.logdet(cov)    
    #loss = 0.5 * (mahalanobis + log_det_cov) - log_jac_det
    loss = 0.5 * mahalanobis - log_jac_det

    return loss


def subnet_conv_func(kernel_size, hidden_ratio):
    def subnet_conv(in_channels, out_channels):
        hidden_channels = int(in_channels * hidden_ratio)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size, padding="same"),
        )

    return subnet_conv


def nf_fast_flow(input_chw, conv3x3_only, hidden_ratio, flow_steps, clamp=2.0):
    nodes = Ff.SequenceINN(*input_chw)
    for i in range(flow_steps):
        if i % 2 == 1 and not conv3x3_only:
            kernel_size = 1
        else:
            kernel_size = 3
        nodes.append(
            Fm.AllInOneBlock,
            subnet_constructor=subnet_conv_func(kernel_size, hidden_ratio),
            affine_clamping=clamp,
            permute_soft=False,
        )
    return nodes


class FastFlow(nn.Module):
    def __init__(
        self,
        backbone_name,
        flow_steps,
        input_size,
        backbone_weights=None,
        conv3x3_only=False,
        hidden_ratio=1.0,
        gmm_values=None,
        in_channels=3,
        out_indices=[1, 2, 3],
        use_proj=False,
    ):
        super(FastFlow, self).__init__()
        assert (
            backbone_name in const.SUPPORTED_BACKBONES
        ), "backbone_name must be one of {}".format(const.SUPPORTED_BACKBONES)

        if backbone_name in [const.BACKBONE_CAIT, const.BACKBONE_DEIT]:
            self.feature_extractor = timm.create_model(backbone_name, pretrained=True, in_chans=in_channels)
            channels = [768]
            scales = [16]
        else:
            self.feature_extractor = timm.create_model(
                backbone_name,
                pretrained=True,
                features_only=True,
                out_indices=out_indices,
                in_chans=in_channels
            )
            if backbone_weights is not None:
                print(f"Loading backbone weights from {backbone_weights}")
                backbone_weights = torch.load(backbone_weights)["model_state_dict"]
                self.feature_extractor.load_state_dict(backbone_weights, strict=False)
                print(f"Backbone weights loaded")
            channels = self.feature_extractor.feature_info.channels()
            scales = self.feature_extractor.feature_info.reduction()

            # for transformers, use their pretrained norm w/o grad
            # for resnets, self.norms are trainable LayerNorm
            self.norms = nn.ModuleList()
            for in_channels, scale in zip(channels, scales):
                self.norms.append(
                    nn.LayerNorm(
                        [in_channels, int(input_size / scale), int(input_size / scale)],
                        elementwise_affine=True,
                    )
                )

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Projection layer: Convolutional network with 1x1 convolutions that maintains feature dimensions
        # Operates on channel dimension, processing each spatial location independently
        self.use_proj = use_proj
        if self.use_proj:
            self.projection_layers = nn.ModuleList()
            for in_channels in channels:
                # Convolutional projection: C -> hidden_dim -> C with 1x1 kernels
                # 1x1 convolutions maintain spatial dimensions while transforming channels
                hidden_dim = int(in_channels / 2)  # Hidden dimension
                self.projection_layers.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=True),  # 1x1 conv
                        nn.ReLU(inplace=True),
                        nn.Conv2d(hidden_dim, in_channels, kernel_size=1, bias=True)  # 1x1 conv
                    )
                )
        else:
            self.projection_layers = None

        self.nf_flows = nn.ModuleList()
        for in_channels, scale in zip(channels, scales):
            self.nf_flows.append(
                nf_fast_flow(
                    [in_channels, int(input_size / scale), int(input_size / scale)],
                    conv3x3_only=conv3x3_only,
                    hidden_ratio=hidden_ratio,
                    flow_steps=flow_steps,
                )
            )
        self.input_size = input_size

        gmm_values = gmm_values["real"]
        self.means = []
        self.covs = []
        translation_param = 0.0
        
        for i in range(len(gmm_values)):
            self.means.append(gmm_values[i][0] + translation_param) 
            self.covs.append(gmm_values[i][1])
        #self.covs = [np.expand_dims(np.eye(cov.shape[1]),axis=0) for cov in self.covs]

    def forward(self, x):
        self.feature_extractor.eval()
        if isinstance(
            self.feature_extractor, timm.models.vision_transformer.VisionTransformer
        ):
            x = self.feature_extractor.patch_embed(x)
            cls_token = self.feature_extractor.cls_token.expand(x.shape[0], -1, -1)
            if self.feature_extractor.dist_token is None:
                x = torch.cat((cls_token, x), dim=1)
            else:
                x = torch.cat(
                    (
                        cls_token,
                        self.feature_extractor.dist_token.expand(x.shape[0], -1, -1),
                        x,
                    ),
                    dim=1,
                )
            x = self.feature_extractor.pos_drop(x + self.feature_extractor.pos_embed)
            for i in range(8):  # paper Table 6. Block Index = 7
                x = self.feature_extractor.blocks[i](x)
            x = self.feature_extractor.norm(x)
            x = x[:, 2:, :]
            N, _, C = x.shape
            x = x.permute(0, 2, 1)
            x = x.reshape(N, C, self.input_size // 16, self.input_size // 16)
            features = [x]
        elif isinstance(self.feature_extractor, timm.models.cait.Cait):
            x = self.feature_extractor.patch_embed(x)
            x = x + self.feature_extractor.pos_embed
            x = self.feature_extractor.pos_drop(x)
            for i in range(41):  # paper Table 6. Block Index = 40
                x = self.feature_extractor.blocks[i](x)
            N, _, C = x.shape
            x = self.feature_extractor.norm(x)
            x = x.permute(0, 2, 1)
            x = x.reshape(N, C, self.input_size // 16, self.input_size // 16)
            features = [x]
        else:
            features = self.feature_extractor(x)
            features = [self.norms[i](feature) for i, feature in enumerate(features)]

        # Apply projection layer (Conv2d) to each feature level if enabled
        # 1x1 convolutions process each spatial location independently: (B, C, H, W) -> (B, C, H, W)
        if self.use_proj:
            projected_features = []
            for i, feature in enumerate(features):
                # Apply 1x1 convolutions directly - no reshape needed
                feature_projected = self.projection_layers[i](feature)  # (B, C, H, W) -> (B, C, H, W)
                projected_features.append(feature_projected)
            features = projected_features

        loss = []
        outputs = []
        log_jac_dets = []
        for i, feature in enumerate(features):
            output, log_jac_det = self.nf_flows[i](features[i])
            mu = self.means[i]
            cov = self.covs[i]
            
            mu = torch.tensor(mu, device=output.device)
            cov = torch.tensor(cov, device=output.device)

            #    loss += torch.mean(
            #         0.5 / det(s)**2 * torch.sum((output-m)**2, dim=(1, 2, 3)) - log_jac_det
            #     )
            output = output.mean((2,3)) #TODO AVG POOLING, MAX POOLING, LIKELIHOOD SPAZIALE (USARE ULTIME FEATURES, INTERPOLAZIONE FEATURES)
        
            #output = output.flatten(2,3).max(-1)[0]

            # if i==2:
            loss.append(gaussian_nll_loss(output=output, mu=mu, cov=cov, log_jac_det=log_jac_det))
        
        outputs.append(output)
        log_jac_dets.append(log_jac_det)
        
        ret = {"loss": torch.stack(loss, dim=1).mean(1)}

        """if not self.training:
            anomaly_map_list = []
            anomaly_prob_list = []
            for (output, log_jac_det) in zip(outputs, log_jac_dets):    
                log_prob = -torch.mean(output**2, dim=1, keepdim=True) * 0.5
                prob = torch.exp(log_prob)
                a_map = F.interpolate(
                    -prob,
                    size=[self.input_size, self.input_size],
                    mode="bilinear",
                    align_corners=False,
                )
                
                anomaly_map_list.append(a_map)
                anomaly_prob_list.append(0.5 * torch.sum(output**2, dim=(1, 2, 3)) - log_jac_det)
                
            anomaly_map_list = torch.stack(anomaly_map_list, dim=-1)
            anomaly_map = torch.mean(anomaly_map_list, dim=-1).flatten(1).sum(1)#.min(1)[0]
            
            ret["anomaly_map"] = torch.stack(anomaly_prob_list).mean(0)#anomaly_map"""

        return ret
