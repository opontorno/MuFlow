import FrEIA.framework as Ff
import FrEIA.modules as Fm
import timm
import torch
import torch.nn as nn

from muflow import constants as const

import numpy as np


def build_backbone(model_name: str, config: dict, device=None):
    """Build a frozen backbone for feature extraction."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_indices = config.get("out_indices", [1, 2, 3])

    if model_name in const.CLIP_BACKBONES:
        backbone = CLIPVisualExtractor(model_name, out_block_indices=out_indices)
        backbone_type = "clip"

    else:
        backbone = timm.create_model(
            config.get("backbone_name", model_name),
            pretrained=True, features_only=True, in_chans=3, out_indices=out_indices
        )
        backbone_type = "cnn"

    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval().to(device)

    return backbone, backbone_type, out_indices, device


class CLIPVisualExtractor(nn.Module):
    def __init__(self, backbone_name: str, out_block_indices: list):
        super().__init__()
        try:
            import clip as openai_clip
        except ImportError:
            raise ImportError(
                "The official OpenAI CLIP package is required. "
                "Install with: pip install git+https://github.com/openai/CLIP.git"
            )
        model_name             = const.CLIP_OPENAI_NAMES[backbone_name]
        clip_model, _          = openai_clip.load(model_name, device="cpu")
        self.visual            = clip_model.visual
        self.out_block_indices = sorted(out_block_indices)
        self.patch_size        = const.CLIP_PATCH_SIZE[backbone_name]
        self.hidden_dim        = const.CLIP_CHANNELS[backbone_name]

        for param in self.visual.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> list:
        """Extract CLIP spatial feature maps."""
        v = self.visual
        B = x.shape[0]

        x = v.conv1(x)
        Hf, Wf = x.shape[2], x.shape[3]
        x = x.reshape(B, x.shape[1], -1).permute(0, 2, 1)

        cls = v.class_embedding.to(x.dtype).unsqueeze(0).unsqueeze(0).expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = x + v.positional_embedding.to(x.dtype)
        x   = v.ln_pre(x)
        x   = x.permute(1, 0, 2)

        features = []
        for i, block in enumerate(v.transformer.resblocks):
            x = block(x)
            if i in self.out_block_indices:
                tokens = x[1:].permute(1, 2, 0)
                feat   = tokens.reshape(B, self.hidden_dim, Hf, Wf)
                features.append(feat)
        return features


def gaussian_nll_loss(output, mu, cov, log_jac_det, pooling_type='mean'):
    """Gaussian negative log-likelihood of flow outputs under a GMM component."""
    B, d = output.shape

    cov_inv = torch.linalg.inv(cov)
    diff = (output - mu).reshape(B, d, 1)
    mahalanobis = torch.matmul(diff.transpose(1, 2), torch.matmul(cov_inv, diff)).squeeze()

    loss = 0.5 * mahalanobis - log_jac_det
    if pooling_type == 'flatten':
        loss = torch.log1p(loss)

    return loss, mahalanobis


def subnet_conv_func(kernel_size, hidden_ratio):
    """Build a subnet constructor for FrEIA coupling blocks."""
    def subnet_conv(in_channels, out_channels):
        hidden_channels = int(in_channels * hidden_ratio)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size, padding="same"),
        )

    return subnet_conv


def nf_fast_flow(input_chw, conv3x3_only, hidden_ratio, flow_steps, clamp=2.0):
    """Build a FastFlow normalizing-flow module for one feature scale."""
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
    ):
        super(FastFlow, self).__init__()
        assert (
            backbone_name in const.SUPPORTED_BACKBONES
        ), "backbone_name must be one of {}".format(const.SUPPORTED_BACKBONES)

        if backbone_name in const.CLIP_BACKBONES:
            if in_channels != 3:
                print(f"[WARNING] CLIP only supports in_channels=3; ignoring in_channels={in_channels}")
            self.backbone_type     = 'clip'
            self.feature_extractor = CLIPVisualExtractor(
                backbone_name, out_block_indices=list(out_indices)
            )
            ch      = const.CLIP_CHANNELS[backbone_name]
            ps      = const.CLIP_PATCH_SIZE[backbone_name]
            num_out = len(out_indices)
            channels = [ch] * num_out
            scales   = [ps] * num_out

        else:
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
        self.input_size    = input_size
        self.pooling_type  = pooling_type
        self._backbone_name = backbone_name

        gmm_values = gmm_values["real"]
        self.means = []
        self.covs = []
        translation_param = 0.0

        for i in range(len(gmm_values)):
            self.means.append(gmm_values[i][0] + translation_param)
            self.covs.append(gmm_values[i][1])

    def process_features(self, features):
        """Run features through the flows and score them against the GMM."""
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

            loss_, maha_ = gaussian_nll_loss(output=output, mu=mu, cov=cov, log_jac_det=log_jac_det, pooling_type=self.pooling_type)
            loss.append(loss_)
            mahalanobis.append(maha_)

        mahalanobis_loss = torch.stack(loss, dim=1).mean(1)

        result = {
            "loss": mahalanobis_loss,
            "mahalanobis": torch.stack(mahalanobis, dim=1).mean(1),
        }
        return result

    def predict(self, image) -> dict:
        """Single-image inference (mean NLL over deterministic patches)."""
        from PIL import Image as _PIL
        from muflow.patch_utils import make_patch_transform, repr_patches

        if isinstance(image, np.ndarray):
            image = _PIL.fromarray(image.astype(np.uint8))
        elif not isinstance(image, _PIL.Image):
            raise TypeError(
                f"image must be PIL.Image or np.ndarray, got {type(image).__name__}"
            )

        norm_mean, norm_std = const.get_norm_stats(self._backbone_name)
        transform = make_patch_transform(norm_mean, norm_std)
        plist = repr_patches(image, self.input_size, const.PATCH_NUM_REPR, const.PATCH_SEED)
        patches = torch.stack([transform(p).float() for p in plist])

        device = next(self.nf_flows[0].parameters()).device
        patches = patches.to(device)
        self.eval()
        with torch.no_grad():
            ret = self.forward(patches)

        return {
            'loss':        float(ret['loss'].mean().item()),
            'mahalanobis': float(ret['mahalanobis'].mean().item()),
        }

    def forward(self, x):
        """Forward pass: backbone features → flows → GMM scoring."""
        self.feature_extractor.eval()
        features = self.feature_extractor(x)
        features = [self.norms[i](f) for i, f in enumerate(features)]
        return self.process_features(features)
