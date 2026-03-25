"""
Feature extraction module using a frozen ResNet-18 backbone.

This module provides utilities to:
  - Build a ResNet-18 model with its final FC layer removed, yielding
    512-dimensional global-average-pooled feature vectors.
  - Preprocess images with standard ImageNet normalisation.
  - Batch-extract features from a list of image paths, skipping load failures.

All feature extraction runs under ``torch.no_grad()`` with the model in
``eval()`` mode to ensure reproducible, gradient-free inference.
"""

import logging

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

logger = logging.getLogger(__name__)

# Standard ImageNet normalisation constants
IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# File extensions treated as images (searched case-insensitively)
SUPPORTED_EXTENSIONS   = (".jpg", ".jpeg", ".png", ".bmp")

# Dimensionality of the ResNet-18 global-average-pool output
FEATURE_DIM  = 512


def build_transform()  :
    """
    Build the standard ImageNet preprocessing transform pipeline.

    The pipeline:
      1. Resize to 224 x 224 (ResNet-18 expected input size).
      2. Convert to a float32 tensor in [0, 1].
      3. Normalise with ImageNet channel mean and standard deviation.

    Returns:
        A ``torchvision.transforms.Compose`` object ready to be applied to
        PIL ``Image`` objects.
    """
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_feature_extractor(device )  :
    """
    Build a frozen ResNet-18 feature extractor.

    Loads the official pretrained ImageNet-1K weights, strips the final
    fully-connected classification layer, and returns the remainder as a
    frozen ``nn.Sequential``.  The architecture up to and including the
    global average-pooling layer produces an output of shape
    ``(batch, 512, 1, 1)``.  Callers should squeeze the spatial dimensions
    to obtain a ``(batch, 512)`` feature matrix.

    All parameters are frozen (``requires_grad=False``) and the module is
    placed in evaluation mode to ensure batch-norm statistics remain fixed.

    Args:
        device: The :class:`torch.device` on which the model will run
                (e.g. ``torch.device("cuda")`` or ``torch.device("cpu")``).

    Returns:
        A frozen ``nn.Sequential`` mapping ``(B, 3, 224, 224) -> (B, 512, 1, 1)``.
    """
    logger.info("Loading pretrained ResNet-18 weights (ImageNet-1K)")
    backbone = models.resnet18(pretrained=True)  # pretrained=True: compat with torchvision 0.12.x (Jetson JetPack 4.6)

    # Drop the final FC layer; keep everything through avgpool
    extractor = nn.Sequential(*list(backbone.children())[:-1])

    for param in extractor.parameters():
        param.requires_grad_(False)

    extractor.eval()
    extractor.to(device)

    logger.info(
        "Feature extractor ready on %s (output dim=%d, frozen)", device, FEATURE_DIM
    )
    return extractor


def load_image(
    path , transform 
)   :
    """
    Load and preprocess a single image file.

    Opens the file with PIL, converts it to RGB (handles greyscale and RGBA
    inputs), and applies *transform*.

    Args:
        path: Filesystem path to the image.
        transform: Preprocessing transform to apply after opening.

    Returns:
        A tuple ``(tensor, success)``.  On success, *tensor* has shape
        ``(3, 224, 224)``.  On failure (missing file, corrupt data, etc.),
        *tensor* is ``None`` and *success* is ``False``.  Failures are
        logged at WARNING level.
    """
    try:
        img = Image.open(path).convert("RGB")
        tensor = transform(img)
        return tensor, True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load image '%s': %s", path, exc)
        return None, False


def find_images(directory )  :
    """
    Recursively discover all supported images under *directory*.

    Searches for files with extensions :data:`SUPPORTED_EXTENSIONS` in both
    lower- and upper-case variants.  Duplicate matches (e.g. from symlinks)
    are deduplicated.

    Args:
        directory: Root directory to search.

    Returns:
        A sorted list of :class:`pathlib.Path` objects.

    Raises:
        ValueError: If *directory* does not exist or is not a directory.
    """
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    paths  = []
    for ext in SUPPORTED_EXTENSIONS:
        paths.extend(directory.rglob(f"*{ext}"))
        paths.extend(directory.rglob(f"*{ext.upper()}"))

    unique = sorted(set(paths))
    logger.info("Found %d image(s) in '%s'", len(unique), directory)
    return unique


def extract_features(
    image_paths ,
    extractor ,
    transform ,
    device ,
    batch_size  = 16,
)   :
    """
    Extract 512-dimensional feature vectors from a list of image files.

    Images are loaded on the CPU (via PIL) and forwarded through *extractor*
    in batches on *device*.  Images that fail to load are silently skipped
    (a WARNING is logged per failure).  The returned array and path list are
    index-aligned: ``features[i]`` corresponds to ``valid_paths[i]``.

    Args:
        image_paths: Ordered list of image file paths.
        extractor: Frozen feature extractor (see :func:`build_feature_extractor`).
        transform: Preprocessing transform (see :func:`build_transform`).
        device: Device for forward passes.
        batch_size: Number of images per forward pass.

    Returns:
        A tuple ``(features, valid_paths)`` where:

        - *features*: ``np.ndarray`` of shape ``(N, 512)``, dtype float32.
        - *valid_paths*: List of :class:`pathlib.Path` objects for the N
          images that were successfully loaded and processed.
    """
    # Pre-load all tensors, collecting only successful loads
    tensors  = []
    tensor_paths  = []

    for path in image_paths:
        tensor, ok = load_image(path, transform)
        if ok:
            tensors.append(tensor)
            tensor_paths.append(path)

    n_loaded = len(tensors)
    n_failed = len(image_paths) - n_loaded
    if n_failed:
        logger.warning("%d image(s) failed to load and will be skipped.", n_failed)

    if n_loaded == 0:
        logger.warning("No images could be loaded from the provided list.")
        return np.empty((0, FEATURE_DIM), dtype=np.float32), []

    n_batches = (n_loaded + batch_size - 1) // batch_size
    logger.info(
        "Extracting features: %d images, batch_size=%d, %d batch(es)",
        n_loaded,
        batch_size,
        n_batches,
    )

    all_features  = []
    with torch.no_grad():
        for batch_idx in range(0, n_loaded, batch_size):
            batch_tensors = tensors[batch_idx : batch_idx + batch_size]
            batch = torch.stack(batch_tensors).to(device)

            # Output shape: (B, 512, 1, 1)
            out = extractor(batch)
            # Squeeze spatial dims -> (B, 512)
            out = out.squeeze(-1).squeeze(-1)
            all_features.append(out.cpu().numpy())

            current_batch = batch_idx // batch_size + 1
            logger.debug("Batch %d/%d done", current_batch, n_batches)

    features = np.concatenate(all_features, axis=0).astype(np.float32)
    logger.info("Feature extraction complete: output shape %s", features.shape)
    return features, tensor_paths
