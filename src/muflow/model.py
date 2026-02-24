import FrEIA.framework as Ff
import FrEIA.modules as Fm
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from muflow import constants as const
from muflow.modules import ProjectionLayer, ConvAutoencoder

import numpy as np


class CLIPVisualExtractor(nn.Module):
    """
    Wraps an open_clip visual encoder and exposes a list of intermediate
    spatial feature maps at user-specified transformer block indices.

    The input tensor is expected to be ImageNet-normalised
    (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]).  The extractor
    internally undoes that and re-applies CLIP normalisation so that the
    rest of the pipeline (dataset, transforms) remains unchanged.

    Returns a list of (B, C, Hf, Wf) tensors, one per requested block.
    """

    def __init__(self, backbone_name: str, out_block_indices: list):
        super().__init__()
        try:
            import open_clip
        except ImportError:
            raise ImportError(
                "open_clip_torch is required for CLIP backbones. "
                "Install with: pip install open-clip-torch"
            )
        clip_model_name, pretrained = const.CLIP_OPENCLIP_NAMES[backbone_name]
        clip_model, _, _ = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=pretrained
        )
        self.visual           = clip_model.visual
        self.out_block_indices= sorted(out_block_indices)
        self.patch_size       = const.CLIP_PATCH_SIZE[backbone_name]
        self.hidden_dim       = const.CLIP_CHANNELS[backbone_name]

        # Buffers for renormalisation: ImageNet → CLIP colour stats
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        imagenet_std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        clip_mean     = torch.tensor(const.CLIP_MEAN).view(1, 3, 1, 1)
        clip_std      = torch.tensor(const.CLIP_STD ).view(1, 3, 1, 1)
        self.register_buffer('imagenet_mean', imagenet_mean)
        self.register_buffer('imagenet_std',  imagenet_std)
        self.register_buffer('clip_mean',     clip_mean)
        self.register_buffer('clip_std',      clip_std)

    def forward(self, x: torch.Tensor) -> list:
        # Undo ImageNet norm, apply CLIP norm
        x = x * self.imagenet_std + self.imagenet_mean   # → [0,1]
        x = (x - self.clip_mean) / self.clip_std

        v = self.visual
        B = x.shape[0]

        # Patch embedding: (B, width, Hf, Wf)
        x = v.conv1(x)
        Hf, Wf = x.shape[2], x.shape[3]
        x = x.reshape(B, x.shape[1], -1).permute(0, 2, 1)   # (B, N, C)

        # CLS token + positional embedding
        cls = v.class_embedding.to(x.dtype).unsqueeze(0).unsqueeze(0).expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)                     # (B, N+1, C)
        x   = x + v.positional_embedding.to(x.dtype)
        x   = v.ln_pre(x)
        x   = x.permute(1, 0, 2)                             # (N+1, B, C)

        features = []
        for i, block in enumerate(v.transformer.resblocks):
            x = block(x)
            if i in self.out_block_indices:
                # Strip CLS, reshape to spatial map
                tokens = x[1:].permute(1, 2, 0)              # (B, C, N)
                feat   = tokens.reshape(B, self.hidden_dim, Hf, Wf)
                features.append(feat)
        return features


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
        use_autoencoder=False,
        ae_hidden_ratio=0.5,
    ):
        super(FastFlow, self).__init__()
        assert (
            backbone_name in const.SUPPORTED_BACKBONES
        ), "backbone_name must be one of {}".format(const.SUPPORTED_BACKBONES)

        if backbone_name in [const.BACKBONE_CAIT, const.BACKBONE_DEIT]:
            # ── Legacy transformer backbones (DeiT / CaiT) ────────────────
            self.backbone_type     = 'cait_deit'
            self.feature_extractor = timm.create_model(backbone_name, pretrained=True, in_chans=in_channels)
            channels = [768]
            scales   = [16]

        elif backbone_name in const.DINO_BACKBONES:
            # ── DINOv2 (timm) ──────────────────────────────────────────────
            if in_channels != 3:
                print(f"[WARNING] DINOv2 only supports in_channels=3; ignoring in_channels={in_channels}")
            self.backbone_type  = 'dino'
            timm_name           = const.DINO_TIMM_NAMES[backbone_name]
            self.feature_extractor = timm.create_model(
                timm_name, pretrained=True, img_size=input_size
            )
            # out_indices are used as transformer block indices to extract
            self.dino_out_blocks = list(out_indices)
            ch      = const.DINO_CHANNELS[backbone_name]
            ps      = const.DINO_PATCH_SIZE[backbone_name]
            num_out = len(out_indices)
            channels = [ch] * num_out
            scales   = [ps] * num_out

        elif backbone_name in const.CLIP_BACKBONES:
            # ── CLIP (open_clip) ───────────────────────────────────────────
            if in_channels != 3:
                print(f"[WARNING] CLIP only supports in_channels=3; ignoring in_channels={in_channels}")
            self.backbone_type     = 'clip'
            # out_indices are used as transformer block indices to extract
            self.feature_extractor = CLIPVisualExtractor(
                backbone_name, out_block_indices=list(out_indices)
            )
            ch      = const.CLIP_CHANNELS[backbone_name]
            ps      = const.CLIP_PATCH_SIZE[backbone_name]
            num_out = len(out_indices)
            channels = [ch] * num_out
            scales   = [ps] * num_out

        else:
            # ── CNN backbones (ResNet, WideResNet, DenseNet …) ─────────────
            self.backbone_type     = 'cnn'
            self.feature_extractor = timm.create_model(
                backbone_name,
                pretrained=True,
                features_only=True,
                out_indices=out_indices,
                in_chans=in_channels
            )
            if backbone_weights is not None:
                print(f"Loading backbone weights from {backbone_weights}")
                bw = torch.load(backbone_weights)["model_state_dict"]
                self.feature_extractor.load_state_dict(bw, strict=False)
                print(f"Backbone weights loaded")
            channels = self.feature_extractor.feature_info.channels()
            scales   = self.feature_extractor.feature_info.reduction()

        # ── Trainable LayerNorms (one per output scale) ────────────────────
        # Applied after feature extraction for all backbone types.
        # For cait_deit the norms are created but deliberately NOT used in
        # forward() to preserve the pretrained ViT normalisation.
        self.norms = nn.ModuleList()
        for ch, sc in zip(channels, scales):
            self.norms.append(
                nn.LayerNorm(
                    [ch, int(input_size / sc), int(input_size / sc)],
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

        # Convolutional autoencoders for feature reconstruction
        self.use_autoencoder = use_autoencoder
        if self.use_autoencoder:
            self.autoencoders = nn.ModuleList()
            for in_ch in channels:
                self.autoencoders.append(
                    ConvAutoencoder(
                        in_channels=in_ch,
                        hidden_ratio=ae_hidden_ratio,
                    )
                )
        else:
            self.autoencoders = None

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
    
    def process_features(self, features, use_projection=True, ae_lambda=0.0):
        """
        Process features through optional modules (projection / autoencoder)
        and normalizing flows.

        Args:
            features: List of feature tensors
            use_projection: Whether to apply projection layers (default: True)
            ae_lambda: Weight for autoencoder reconstruction loss (default: 0.0)

        Returns:
            Dictionary with 'loss', 'mahalanobis', and optionally 'recon_loss'
        """
        # ----- Projection layers (contrastive mode) -----
        projected_features = None
        if self.use_proj_layer and use_projection and self.projection_layers is not None:
            projected_features = []
            for i, feature in enumerate(features):
                proj_feat = self.projection_layers[i](feature)
                projected_features.append(proj_feat)
            features = projected_features

        # ----- Convolutional Autoencoder -----
        recon_loss_total = None
        if self.use_autoencoder and self.autoencoders is not None:
            recon_losses = []
            reconstructed = []
            for i, feature in enumerate(features):
                x_hat = self.autoencoders[i](feature)
                recon_losses.append(F.mse_loss(x_hat, feature))
                reconstructed.append(x_hat)
            features = reconstructed  # NF sees reconstructed features
            recon_loss_total = torch.stack(recon_losses).mean()

        # ----- Normalizing Flows -----
        loss = []
        mahalanobis = []
        for i, feature in enumerate(features):
            output, log_jac_det = self.nf_flows[i](feature)
            mu = torch.tensor(self.means[i], device=output.device)
            cov = torch.tensor(self.covs[i], device=output.device)

            if self.pooling_type == 'mean':
                output = output.mean((2, 3))
            elif self.pooling_type == 'max':
                output = output.flatten(2).max(-1)[0]
            elif self.pooling_type == 'mean_std':
                mean_output = output.mean((2, 3))
                std_output = output.std((2, 3))
                output = torch.cat([mean_output, std_output], dim=1)
            elif self.pooling_type == 'flatten':
                output = output.mean(-1).flatten(1)
            else:
                output = output.flatten(1)

            loss_, maha_ = gaussian_nll_loss(output=output, mu=mu, cov=cov, log_jac_det=log_jac_det)
            loss.append(loss_)
            mahalanobis.append(maha_)

        mahalanobis_loss = torch.stack(loss, dim=1).mean(1)  # (B,)

        # ----- Combine losses -----
        if recon_loss_total is not None and ae_lambda > 0.0:
            combined_loss = mahalanobis_loss + ae_lambda * recon_loss_total
        else:
            combined_loss = mahalanobis_loss

        result = {
            "loss": combined_loss,
            "mahalanobis": torch.stack(mahalanobis, dim=1).mean(1),
        }
        if recon_loss_total is not None:
            result["recon_loss"] = recon_loss_total
        return result

    def forward(self, x, noise_std=0.0, adversarial_mode='pure_only', ae_lambda=0.0):
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

        if self.backbone_type == 'cait_deit':
            # ── DeiT ──────────────────────────────────────────────────────
            if isinstance(self.feature_extractor, timm.models.vision_transformer.VisionTransformer):
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
            # ── CaiT ──────────────────────────────────────────────────────
            else:
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

        elif self.backbone_type == 'dino':
            # ── DINOv2 — get_intermediate_layers returns (B, C, H, W) ────
            features = self.feature_extractor.get_intermediate_layers(
                x, n=self.dino_out_blocks, reshape=True
            )
            features = [self.norms[i](f) for i, f in enumerate(features)]

        elif self.backbone_type == 'clip':
            # ── CLIP — CLIPVisualExtractor handles renorm + spatial maps ─
            features = self.feature_extractor(x)
            features = [self.norms[i](f) for i, f in enumerate(features)]

        else:
            # ── CNN (ResNet, etc.) ────────────────────────────────────────
            features = self.feature_extractor(x)
            features = [self.norms[i](feature) for i, feature in enumerate(features)]
        
        # Adversarial training logic
        if adversarial_mode == 'same_batch':
            noisy_features = self.add_gaussian_noise(features, noise_std)
            ret_pure = self.process_features(features, ae_lambda=ae_lambda)
            ret_noisy = self.process_features(noisy_features, ae_lambda=ae_lambda)
            return {
                "loss": ret_pure["loss"],
                "loss_pure": ret_pure["loss"],
                "loss_adv": ret_noisy["loss"],
                "mahalanobis": ret_pure["mahalanobis"],
            }

        elif adversarial_mode == 'noisy_only':
            noisy_features = self.add_gaussian_noise(features, noise_std)
            return self.process_features(noisy_features, ae_lambda=ae_lambda)

        else:  # 'pure_only' or standard forward
            return self.process_features(features, ae_lambda=ae_lambda)
    
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

    def freeze_autoencoders(self):
        """Freeze autoencoder layers"""
        if self.autoencoders is not None:
            for ae in self.autoencoders:
                for param in ae.parameters():
                    param.requires_grad = False
            print("Autoencoders frozen")

    def unfreeze_autoencoders(self):
        """Unfreeze autoencoder layers"""
        if self.autoencoders is not None:
            for ae in self.autoencoders:
                for param in ae.parameters():
                    param.requires_grad = True
            print("Autoencoders unfrozen")

