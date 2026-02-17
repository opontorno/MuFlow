import FrEIA.framework as Ff
import FrEIA.modules as Fm
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import constants as const
from proj_layer import ProjectionLayer

import numpy as np
import pdb


def gaussian_nll_loss(output, mu, cov, log_jac_det):
    
    B, d = output.shape
    
    cov_inv = torch.linalg.inv(cov)
    diff = (output - mu).reshape(B, d, 1) # Shape: (B, d, 1)  TODO: controllare shape output
    mahalanobis = torch.matmul(diff.transpose(1, 2), torch.matmul(cov_inv, diff)).squeeze() # Mahalanobis distance: (x - mu)^T Σ^{-1} (x - mu)

    loss = torch.log1p(0.5 * mahalanobis - log_jac_det)

    return loss, mahalanobis


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
        pooling_type='mean',
        use_adversarial=False,
        noise_differentiable=False,
        use_proj_layer=False,
        proj_hidden_ratio=0.5,
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

        # Projection layers for contrastive learning
        self.use_proj_layer = use_proj_layer
        if self.use_proj_layer:
            self.projection_layers = nn.ModuleList()
            for in_channels in channels:
                proj_layer = ProjectionLayer(
                    in_channels=in_channels,
                    hidden_ratio=proj_hidden_ratio
                )
                self.projection_layers.append(proj_layer)
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
        self.pooling_type = pooling_type
        
        # Adversarial training config
        self.use_adversarial = use_adversarial
        self.noise_differentiable = noise_differentiable

        gmm_values = gmm_values["real"]
        self.means = []
        self.covs = []
        translation_param = 0.0
        
        for i in range(len(gmm_values)):
            self.means.append(gmm_values[i][0] + translation_param) 
            self.covs.append(gmm_values[i][1])
        #self.covs = [np.expand_dims(np.eye(cov.shape[1]),axis=0) for cov in self.covs]
    
    def add_gaussian_noise(self, features, noise_std):
        """
        Add Gaussian noise to features, scaled relative to feature magnitude.
        
        Args:
            features: List of feature tensors from backbone
            noise_std: Relative standard deviation multiplier (e.g., 0.1 = 10% of feature std)
        
        Returns:
            List of noisy feature tensors
        """
        noisy_features = []
        for feature in features:
            feature_std = feature.std(dim=[0, 2, 3], keepdim=True) + 1e-8
            noise = torch.randn_like(feature) * noise_std * feature_std
            
            if not self.noise_differentiable:
                noise = noise.detach()
            
            noisy_feature = feature + noise
            noisy_features.append(noisy_feature)
        
        return noisy_features
    
    def process_features(self, features, use_projection=True):
        """
        Process features through projection layers (if enabled) and normalizing flows.
        
        Args:
            features: List of feature tensors
            use_projection: Whether to apply projection layers (default: True)
        
        Returns:
            Dictionary with 'loss', 'mahalanobis', and optionally 'projected_features'
        """
        # Apply projection layers if enabled
        projected_features = None
        if self.use_proj_layer and use_projection and self.projection_layers is not None:
            projected_features = []
            for i, feature in enumerate(features):
                proj_feat = self.projection_layers[i](feature)
                projected_features.append(proj_feat)
            features = projected_features
        
        loss = []
        mahalanobis = []
        for i, feature in enumerate(features):
            output, log_jac_det = self.nf_flows[i](feature)
            mu = self.means[i]
            cov = self.covs[i]
            
            mu = torch.tensor(mu, device=output.device)
            cov = torch.tensor(cov, device=output.device)

            if self.pooling_type == 'mean':
                output = output.mean((2,3))
            elif self.pooling_type == 'max':
                output = output.flatten(2).max(-1)[0]
            elif self.pooling_type == 'mean_std':
                mean_output = output.mean((2,3))
                std_output = output.std((2,3))
                output = torch.cat([mean_output, std_output], dim=1)
            elif self.pooling_type == 'flatten':
                output = output.mean(-1).flatten(1)
            else:
                output = output.flatten(1)

            loss_, maha_ = gaussian_nll_loss(output=output, mu=mu, cov=cov, log_jac_det=log_jac_det)
            loss.append(loss_)
            mahalanobis.append(maha_)
        
        return {
            "loss": torch.stack(loss, dim=1).mean(1),
            "mahalanobis": torch.stack(mahalanobis, dim=1).mean(1)
        }

    def forward(self, x, noise_std=0.0, adversarial_mode='pure_only'):
        """
        Forward pass with optional adversarial training.
        
        Args:
            x: Input tensor
            noise_std: Standard deviation of Gaussian noise (default: 0.0)
            adversarial_mode: Mode for adversarial training
                - 'pure_only': Only process pure features (default)
                - 'noisy_only': Only process noisy features
                - 'same_batch': Process both pure and noisy features
        
        Returns:
            Dictionary with loss and mahalanobis distance
            If adversarial_mode='same_batch', returns loss_pure and loss_adv
        """
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
        
        # Adversarial training logic
        if adversarial_mode == 'same_batch':
            # Process both pure and noisy features
            noisy_features = self.add_gaussian_noise(features, noise_std)
            
            # Process pure features
            ret_pure = self.process_features(features)
            
            # Process noisy features
            ret_noisy = self.process_features(noisy_features)
            
            return {
                "loss": ret_pure["loss"],  # For compatibility
                "loss_pure": ret_pure["loss"],
                "loss_adv": ret_noisy["loss"],
                "mahalanobis": ret_pure["mahalanobis"],
            }
        
        elif adversarial_mode == 'noisy_only':
            # Only process noisy features
            noisy_features = self.add_gaussian_noise(features, noise_std)
            return self.process_features(noisy_features)
        
        else:  # 'pure_only' or standard forward
            # Only process pure features
            return self.process_features(features)
    
    def freeze_projection_layers(self):
        """Freeze projection layers for training FastFlow"""
        if self.projection_layers is not None:
            for proj_layer in self.projection_layers:
                for param in proj_layer.parameters():
                    param.requires_grad = False
            print("Projection layers frozen")
    
    def unfreeze_projection_layers(self):
        """Unfreeze projection layers for training them"""
        if self.projection_layers is not None:
            for proj_layer in self.projection_layers:
                for param in proj_layer.parameters():
                    param.requires_grad = True
            print("Projection layers unfrozen")
    
    def freeze_fastflow(self):
        """Freeze FastFlow (normalizing flows) for training projection layers"""
        for nf_flow in self.nf_flows:
            for param in nf_flow.parameters():
                param.requires_grad = False
        print("FastFlow frozen")
    
    def unfreeze_fastflow(self):
        """Unfreeze FastFlow for training"""
        for nf_flow in self.nf_flows:
            for param in nf_flow.parameters():
                param.requires_grad = True
        print("FastFlow unfrozen")

