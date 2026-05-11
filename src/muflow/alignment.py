"""
MuFlow — Face alignment pipelines.

Implements two canonical face alignment methods:
  'ffhq'     : FFHQ-style quad-transform (Karras et al., NVlabs/ffhq-dataset)
  'celeba_hq': CelebA-HQ-style 5-point similarity transform (Liu et al., 2015)

Both pipelines use face-alignment (Adrian Bulat) with FAN + S3FD detector
for 68-point landmark detection.
"""
import cv2
import numpy as np
from PIL import Image
from typing import Optional


# ── CelebA canonical 5-point positions for a 112×112 output ───────────────
# Scale by (output_size / 112) for other resolutions.
# Source: InsightFace / ArcFace reference alignment (Liu et al., 2015).
CELEBA_LANDMARKS_REF_112 = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # left mouth corner
    [70.7299, 92.2041],   # right mouth corner
], dtype=np.float32)


class FaceNotFoundError(Exception):
    """Raised when no face is detected in the image."""
    pass


class Aligner:
    """
    Face aligner with two alignment modes.

    Args:
        mode        : 'ffhq' (default) or 'celeba_hq'.
        output_size : output square side in pixels (default 256).
        device      : 'cuda' or 'cpu' for the landmark detector.
                      When None (default) CUDA is used if available.
    """

    def __init__(self, mode: str = 'ffhq', output_size: Optional[int] = None,
                 device: Optional[str] = None):
        """
        Args:
            output_size : output square side in pixels.
                          Pass None to preserve the input image dimensions
                          (the output will be square with side = max(W, H)).
        """
        if mode not in ('ffhq', 'celeba_hq'):
            raise ValueError(
                f"mode must be 'ffhq' or 'celeba_hq', got '{mode}'"
            )
        self.mode        = mode
        self.output_size = output_size   # None → resolved per-image in align()
        self._fa         = self._load_detector(device)

    # ── detector ────────────────────────────────────────────────────────────

    def _load_detector(self, device: Optional[str]):
        try:
            import face_alignment
        except ImportError:
            raise ImportError(
                "face-alignment is not installed.\n"
                "  Run: pip install face-alignment"
            )
        try:
            lm_type = face_alignment.LandmarksType.TWO_D
        except AttributeError:
            lm_type = face_alignment.LandmarksType._2D   # older API

        if device is None:
            try:
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                device = 'cpu'

        return face_alignment.FaceAlignment(
            lm_type, flip_input=False, device=device,
        )

    def _detect_landmarks(self, img_np: np.ndarray) -> np.ndarray:
        """
        Detect 68 2D facial landmarks on a uint8 RGB image.

        Args:
            img_np : (H, W, 3) uint8 RGB array.

        Returns:
            (68, 2) float32 landmark array.

        Raises:
            FaceNotFoundError if no face is detected or the detector fails.
        """
        try:
            preds = self._fa.get_landmarks(img_np)
        except Exception as exc:
            raise FaceNotFoundError(
                f"Landmark detector raised an exception: {exc}"
            ) from exc

        if preds is None or len(preds) == 0:
            raise FaceNotFoundError("No face detected in the image.")

        return np.array(preds[0], dtype=np.float32)

    # ── public API ───────────────────────────────────────────────────────────

    def align(self, image: Image.Image) -> Image.Image:
        """
        Align a PIL RGB image to the canonical face position.

        Args:
            image : PIL.Image.Image (any mode — converted to RGB internally).

        Returns:
            Aligned PIL.Image.Image.
            Size is (output_size × output_size) when output_size is set, or
            (max(W,H) × max(W,H)) when output_size=None (preserves input scale).

        Raises:
            FaceNotFoundError if no face is detected or the warp fails.
        """
        img_rgb  = image.convert('RGB')
        img_np   = np.array(img_rgb)
        lm68     = self._detect_landmarks(img_np)
        out_size = self.output_size if self.output_size is not None \
                   else max(img_rgb.size)

        if self.mode == 'ffhq':
            return self._align_ffhq(img_rgb, lm68, out_size)
        return self._align_celeba(img_rgb, lm68, out_size)

    # ── FFHQ alignment ──────────────────────────────────────────────────────

    def _align_ffhq(self, img_pil: Image.Image, lm68: np.ndarray,
                    out_size: int) -> Image.Image:
        """
        FFHQ-style alignment via oriented quad-transform.

        Faithfully adapted from:
        https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py
        (Karras et al., 2019)

        The output includes the full face context (hair, forehead, chin) at
        output_size × output_size with eyes at ≈35% from the top.
        Renders at 2× resolution then downsamples (good quality/speed trade-off).
        Padding uses simple reflect fill instead of Gaussian smear for speed.
        """
        output_size    = out_size
        transform_size = output_size * 2   # 2× oversampling then downsample
        enable_padding = True

        lm = lm68.astype(np.float64)

        eye_left     = lm[36:42].mean(0)
        eye_right    = lm[42:48].mean(0)
        eye_avg      = (eye_left + eye_right) * 0.5
        eye_to_eye   = eye_right - eye_left
        mouth_left   = lm[48]
        mouth_right  = lm[54]
        mouth_avg    = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Oriented crop axes
        x  = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y  = np.flipud(x) * [-1, 1]
        c  = eye_avg + eye_to_mouth * 0.1
        quad  = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        img = img_pil.copy()

        # ── Shrink ──
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(img.size[0] / shrink)),
                     int(np.rint(img.size[1] / shrink)))
            img   = img.resize(rsize, Image.LANCZOS)
            quad /= shrink
            qsize /= shrink

        # ── Crop ──
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:, 0]))),
                int(np.floor(min(quad[:, 1]))),
                int(np.ceil(max(quad[:, 0]))),
                int(np.ceil(max(quad[:, 1]))))
        crop = (max(crop[0] - border, 0),
                max(crop[1] - border, 0),
                min(crop[2] + border, img.size[0]),
                min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img   = img.crop(crop)
            quad -= crop[0:2]

        # ── Pad ──
        pad = (int(np.floor(min(quad[:, 0]))),
               int(np.floor(min(quad[:, 1]))),
               int(np.ceil(max(quad[:, 0]))),
               int(np.ceil(max(quad[:, 1]))))
        pad = (max(-pad[0] + border, 0),
               max(-pad[1] + border, 0),
               max(pad[2] - img.size[0] + border, 0),
               max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            # Simple reflect padding — much faster than the Gaussian smear in
            # the original FFHQ script; quality difference is negligible at
            # 256px output resolution.
            img_arr = np.array(img, dtype=np.uint8)
            img_arr = np.pad(img_arr,
                             ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)),
                             mode='reflect')
            img      = Image.fromarray(img_arr)
            quad    += pad[:2]

        # ── Warp ──
        img = img.transform(
            (transform_size, transform_size),
            Image.QUAD,
            (quad + 0.5).flatten(),
            Image.BILINEAR,
        )
        if output_size < transform_size:
            img = img.resize((output_size, output_size), Image.LANCZOS)

        return img

    # ── CelebA-HQ alignment ─────────────────────────────────────────────────

    def _align_celeba(self, img_pil: Image.Image, lm68: np.ndarray,
                      out_size: int) -> Image.Image:
        """
        CelebA-HQ-style 5-point similarity-transform alignment (Liu et al., 2015).

        Extracts 5 canonical points from the 68 FAN landmarks, then computes
        a similarity transform (scale + rotation + translation, no shear) that
        maps them to the CelebA canonical positions scaled to output_size.
        """
        lm = lm68.astype(np.float32)

        src_pts = np.array([
            lm[36:42].mean(0),   # left eye  (mean of pts 36–41)
            lm[42:48].mean(0),   # right eye (mean of pts 42–47)
            lm[30],              # nose tip
            lm[48],              # left mouth corner
            lm[54],              # right mouth corner
        ], dtype=np.float32)

        dst_pts = CELEBA_LANDMARKS_REF_112 * (out_size / 112.0)

        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
        if M is None:
            raise FaceNotFoundError(
                "Could not estimate affine transform from detected landmarks."
            )

        warped = cv2.warpAffine(
            np.array(img_pil), M,
            (out_size, out_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        return Image.fromarray(warped)
