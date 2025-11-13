#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BIDS-conform image registration utilities for neuroimaging pipelines.

Provides functions for:
- Two-stage registration (rigid + SyN)
- ROI transformation and alignment
- Brain mask creation and filtering
- BIDS-compliant output naming

BIDS Naming Convention for Derivatives:
    sub-<label>[_ses-<label>][_space-<label>][_desc-<label>][_suffix].nii.gz

Examples:
    sub-01_space-MNI152NLin2009cAsym_T1w.nii.gz
    sub-01_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz
    sub-01_space-MNI152NLin2009cAsym_desc-syn_T1w.nii.gz
"""

import logging
import os
from pathlib import Path
from typing import Tuple, Optional, Dict, Union
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import ants
from scipy import ndimage

logger = logging.getLogger(__name__)


@dataclass
class RegistrationResult:
    """Data class for BIDS-conform registration results."""
    registered_image: Path
    registered_roi: Optional[Path] = None
    normalized_image: Optional[Path] = None
    brain_mask: Optional[Path] = None
    roi_mask_brain_filtered: Optional[Path] = None
    rigid_image: Optional[Path] = None


@dataclass
class RegistrationParams:
    """Data class for registration parameters."""
    # Rigid registration parameters
    rigid_metric: str = "mattes"
    rigid_grad_step: float = 0.1
    
    # SyN registration parameters
    syn_grad_step: float = 0.1
    syn_iterations: int = 10
    
    # Brain mask parameters
    brain_mask_threshold: float = 0.1
    roi_mask_threshold: float = 0.05
    
    # Processing parameters
    normalize: bool = True
    create_brain_mask: bool = True
    filter_roi_by_brain: bool = True
    skip_if_exists: bool = True


@dataclass
class BIDSNamingConfig:
    """Configuration for BIDS-compliant output naming."""
    subject_id: str          # e.g., "01"
    session_id: Optional[str] = None  # e.g., "01" for ses-01
    space: str = "MNI152NLin2009cAsym"  # Standard space
    modality: str = "T2w"    # Modality suffix (T1w, T2w, FLAIR, etc)
    derivatives_dir: Optional[Path] = None


class BIDSNamer:
    """Generate BIDS-compliant filenames for registration outputs."""
    
    @staticmethod
    def _build_basename(config: BIDSNamingConfig) -> str:
        """Build BIDS base filename without suffix."""
        basename = f"sub-{config.subject_id}"
        
        if config.session_id:
            basename += f"_ses-{config.session_id}"
        
        basename += f"_space-{config.space}"
        
        return basename
    
    @staticmethod
    def registered_image(config: BIDSNamingConfig, description: str = "syn") -> str:
        """
        BIDS filename for registered image.
        
        Example: sub-01_space-MNI152NLin2009cAsym_desc-syn_T2w.nii.gz
        """
        basename = BIDSNamer._build_basename(config)
        return f"{basename}_desc-{description}_{config.modality}.nii.gz"
    
    @staticmethod
    def rigid_image(config: BIDSNamingConfig) -> str:
        """
        BIDS filename for rigid-registered image.
        
        Example: sub-01_space-MNI152NLin2009cAsym_desc-rigid_T2w.nii.gz
        """
        basename = BIDSNamer._build_basename(config)
        return f"{basename}_desc-rigid_{config.modality}.nii.gz"
    
    @staticmethod
    def normalized_image(config: BIDSNamingConfig) -> str:
        """
        BIDS filename for normalized image.
        
        Example: sub-01_space-MNI152NLin2009cAsym_desc-synNorm_T2w.nii.gz
        """
        basename = BIDSNamer._build_basename(config)
        return f"{basename}_desc-synNorm_{config.modality}.nii.gz"
    
    @staticmethod
    def brain_mask(config: BIDSNamingConfig) -> str:
        """
        BIDS filename for brain mask.
        
        Example: sub-01_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz
        """
        basename = BIDSNamer._build_basename(config)
        return f"{basename}_desc-brain_mask.nii.gz"
    
    @staticmethod
    def roi_mask(config: BIDSNamingConfig, roi_name: str = "lesion") -> str:
        """
        BIDS filename for ROI mask.
        
        Example: sub-01_space-MNI152NLin2009cAsym_desc-lesion_mask.nii.gz
        """
        basename = BIDSNamer._build_basename(config)
        return f"{basename}_desc-{roi_name}_mask.nii.gz"
    
    @staticmethod
    def roi_mask_brain_filtered(config: BIDSNamingConfig, roi_name: str = "lesion") -> str:
        """
        BIDS filename for brain-filtered ROI mask.
        
        Example: sub-01_space-MNI152NLin2009cAsym_desc-lesionBrainFiltered_mask.nii.gz
        """
        basename = BIDSNamer._build_basename(config)
        return f"{basename}_desc-{roi_name}BrainFiltered_mask.nii.gz"
    
    @staticmethod
    def rigid_roi_mask(config: BIDSNamingConfig, roi_name: str = "lesion") -> str:
        """
        BIDS filename for rigid-registered ROI mask.
        
        Example: sub-01_space-MNI152NLin2009cAsym_desc-lesionRigid_mask.nii.gz
        """
        basename = BIDSNamer._build_basename(config)
        return f"{basename}_desc-{roi_name}Rigid_mask.nii.gz"
    
    @staticmethod
    def syn_roi_mask(config: BIDSNamingConfig, roi_name: str = "lesion") -> str:
        """
        BIDS filename for SyN-registered ROI mask.
        
        Example: sub-01_space-MNI152NLin2009cAsym_desc-lesionSyn_mask.nii.gz
        """
        basename = BIDSNamer._build_basename(config)
        return f"{basename}_desc-{roi_name}Syn_mask.nii.gz"


# ============================================================================
# REGISTRATION UTILITIES
# ============================================================================

def _validate_registration_inputs(
    img_path: Union[str, Path],
    standard_space_path: Union[str, Path],
    roi_path: Optional[Union[str, Path]] = None,
    verbose: bool = False
) -> Tuple[bool, str]:
    """
    Validate registration input files.
    
    Parameters:
        img_path: Path to moving image.
        standard_space_path: Path to fixed (standard) image.
        roi_path: Optional path to ROI image.
        verbose: If True, log validation details.
    
    Returns:
        Tuple[bool, str]: (is_valid, message)
    """
    img_path = Path(img_path)
    standard_space_path = Path(standard_space_path)
    
    if not img_path.exists():
        return False, f"Moving image not found: {img_path}"
    
    if not standard_space_path.exists():
        return False, f"Standard space image not found: {standard_space_path}"
    
    if roi_path:
        roi_path = Path(roi_path)
        if not roi_path.exists():
            if verbose:
                logger.warning(f"ROI path provided but not found: {roi_path}")
            return True, "Warning: ROI not found"
    
    if verbose:
        logger.info(f"✓ Moving image: {img_path}")
        logger.info(f"✓ Standard space: {standard_space_path}")
        if roi_path and roi_path.exists():
            logger.info(f"✓ ROI image: {roi_path}")
    
    return True, "Inputs valid"


def _create_brain_mask_from_normalized(
    normalized_image: np.ndarray,
    threshold: float = 0.1,
    min_component_size: int = 100
) -> np.ndarray:
    """
    Create brain mask from normalized image using connected components.

    Parameters:
        normalized_image: Normalized image array (0-1 range).
        threshold: Intensity threshold for brain tissue.
        min_component_size: Minimum voxels for valid component.

    Returns:
        np.ndarray: Binary brain mask.
    """
    try:
        brain_mask_threshold = normalized_image > threshold
        labeled_array, num_features = ndimage.label(brain_mask_threshold)
        
        if num_features == 0:
            logger.warning("No brain tissue found, using threshold mask")
            return brain_mask_threshold.astype(np.uint8)
        
        component_sizes = np.bincount(labeled_array.ravel())
        component_sizes[0] = 0
        
        largest_component = np.argmax(component_sizes)
        
        if component_sizes[largest_component] < min_component_size:
            logger.warning(
                f"Largest component only {component_sizes[largest_component]} voxels, "
                f"using all components above threshold"
            )
            return brain_mask_threshold.astype(np.uint8)
        
        brain_mask = (labeled_array == largest_component).astype(np.uint8)
        logger.info(
            f"Brain mask created: {np.sum(brain_mask)} voxels "
            f"({component_sizes[largest_component]} in largest component)"
        )
        
        return brain_mask
        
    except Exception as e:
        logger.error(f"Error creating brain mask: {e}")
        raise


def _normalize_image(image_array: np.ndarray) -> np.ndarray:
    """
    Normalize image intensities to 0-1 range.

    Parameters:
        image_array: Input image array.

    Returns:
        np.ndarray: Normalized image (0-1 range, float32).
    """
    mask = image_array > 0
    normalized = np.zeros_like(image_array, dtype=np.float32)
    
    if not np.any(mask):
        logger.warning("Image contains no positive values")
        return normalized
    
    data_min = np.min(image_array[mask])
    data_max = np.max(image_array[mask])
    
    if data_min == data_max:
        logger.warning("Image has constant intensity")
        normalized[mask] = 1.0
    else:
        normalized[mask] = (image_array[mask] - data_min) / (data_max - data_min)
    
    return normalized


# ============================================================================
# RIGID REGISTRATION
# ============================================================================

def register_rigid(
    fixed_image: 'ants.ANTsImage',
    moving_image: 'ants.ANTsImage',
    metric: str = "mattes",
    grad_step: float = 0.1,
    verbose: bool = False
) -> Dict:
    """
    Perform rigid registration (rotation + translation).

    Parameters:
        fixed_image: Fixed (target) ANTsImage.
        moving_image: Moving (source) ANTsImage.
        metric: Metric for registration ("mattes" or "meansquares").
        grad_step: Gradient step size for optimization.
        verbose: If True, show ANTs verbose output.

    Returns:
        Dict: Registration output from ants.registration.
    """
    try:
        logger.info(f"Performing rigid registration (metric={metric}, grad_step={grad_step})")
        
        tx_rigid = ants.registration(
            fixed=fixed_image,
            moving=moving_image,
            type_of_transform="Rigid",
            aff_metric=metric,
            grad_step=grad_step,
            verbose=verbose
        )
        
        logger.info("✓ Rigid registration completed")
        return tx_rigid
        
    except Exception as e:
        logger.error(f"Rigid registration failed: {e}", exc_info=True)
        raise RuntimeError(f"Rigid registration failed: {e}") from e


def apply_rigid_transform(
    fixed_image: 'ants.ANTsImage',
    moving_image: 'ants.ANTsImage',
    transforms: list,
    interpolator: str = "linear"
) -> 'ants.ANTsImage':
    """
    Apply rigid transformation to image.

    Parameters:
        fixed_image: Fixed (target) ANTsImage.
        moving_image: Moving (source) ANTsImage.
        transforms: List of transformation files.
        interpolator: Interpolation method ("linear", "nearestNeighbor", etc).

    Returns:
        ANTsImage: Transformed image.
    """
    try:
        transformed = ants.apply_transforms(
            fixed=fixed_image,
            moving=moving_image,
            transformlist=transforms,
            interpolator=interpolator
        )
        return transformed
        
    except Exception as e:
        logger.error(f"Failed to apply rigid transform: {e}", exc_info=True)
        raise RuntimeError(f"Transform application failed: {e}") from e


# ============================================================================
# SYN REGISTRATION
# ============================================================================

def register_syn(
    fixed_image: 'ants.ANTsImage',
    moving_image: 'ants.ANTsImage',
    grad_step: float = 0.1,
    iterations: int = 10,
    verbose: bool = False
) -> Dict:
    """
    Perform SyN registration (nonlinear/deformable).

    Parameters:
        fixed_image: Fixed (target) ANTsImage.
        moving_image: Moving (source) ANTsImage.
        grad_step: Gradient step size for optimization.
        iterations: Number of iterations.
        verbose: If True, show ANTs verbose output.

    Returns:
        Dict: Registration output from ants.registration.
    """
    try:
        logger.info(f"Performing SyN registration (iterations={iterations}, grad_step={grad_step})")
        
        tx_syn = ants.registration(
            fixed=fixed_image,
            moving=moving_image,
            type_of_transform="SyN",
            grad_step=grad_step,
            number_of_iterations=iterations,
            verbose=verbose
        )
        
        logger.info("✓ SyN registration completed")
        return tx_syn
        
    except Exception as e:
        logger.error(f"SyN registration failed: {e}", exc_info=True)
        raise RuntimeError(f"SyN registration failed: {e}") from e


# ============================================================================
# MAIN TWO-STAGE REGISTRATION (BIDS-CONFORM)
# ============================================================================

def register_to_standard_space(
    img_path: Union[str, Path],
    standard_space_path: Union[str, Path],
    output_dir: Union[str, Path],
    bids_config: BIDSNamingConfig,
    roi_path: Optional[Union[str, Path]] = None,
    roi_name: str = "lesion",
    params: Optional[RegistrationParams] = None,
    verbose: bool = False
) -> RegistrationResult:
    """
    Perform BIDS-conform two-stage registration (rigid + SyN).
    
    Uses BIDS (Brain Imaging Data Structure) convention for output naming.

    Parameters:
        img_path: Path to moving (source) image.
        standard_space_path: Path to fixed (standard space) image.
        output_dir: Directory to save outputs.
        bids_config: BIDSNamingConfig with subject/session/space information.
        roi_path: Optional path to ROI image.
        roi_name: Name for ROI in BIDS filename (e.g., "lesion", "wmh").
        params: RegistrationParams with custom parameters (uses defaults if None).
        verbose: If True, enable verbose logging.

    Returns:
        RegistrationResult: Dataclass with all output paths.

    Raises:
        FileNotFoundError: If input files don't exist.
        RuntimeError: If registration fails.
        
    Example:
        >>> bids_config = BIDSNamingConfig(
        ...     subject_id="01",
        ...     session_id="01",
        ...     space="MNI152NLin2009cAsym",
        ...     modality="FLAIR"
        ... )
        >>> result = register_to_standard_space(
        ...     img_path="sub-01_ses-01_FLAIR.nii.gz",
        ...     standard_space_path="mni152.nii.gz",
        ...     output_dir="derivatives/registration",
        ...     bids_config=bids_config,
        ...     roi_path="sub-01_ses-01_lesions.nii.gz"
        ... )
        >>> print(result.registered_image)
        # sub-01_ses-01_space-MNI152NLin2009cAsym_desc-syn_FLAIR.nii.gz
    """
    # Convert paths
    img_path = Path(img_path)
    standard_space_path = Path(standard_space_path)
    output_dir = Path(output_dir)
    roi_path = Path(roi_path) if roi_path else None

    # Use default parameters if not provided
    if params is None:
        params = RegistrationParams()

    # Validate inputs
    is_valid, msg = _validate_registration_inputs(
        img_path, standard_space_path, roi_path, verbose=verbose
    )
    if not is_valid:
        raise FileNotFoundError(msg)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get BIDS-conform filenames
    rigid_filename = BIDSNamer.rigid_image(bids_config)
    syn_filename = BIDSNamer.registered_image(bids_config, description="syn")
    normalized_filename = BIDSNamer.normalized_image(bids_config) if params.normalize else None
    brain_mask_filename = BIDSNamer.brain_mask(bids_config) if params.create_brain_mask else None
    rigid_roi_filename = BIDSNamer.rigid_roi_mask(bids_config, roi_name) if roi_path else None
    syn_roi_filename = BIDSNamer.syn_roi_mask(bids_config, roi_name) if roi_path else None
    roi_brain_filtered_filename = BIDSNamer.roi_mask_brain_filtered(bids_config, roi_name) if (roi_path and params.filter_roi_by_brain) else None

    # Define output paths
    rigid_output = output_dir / rigid_filename
    syn_output = output_dir / syn_filename
    syn_roi_output = output_dir / syn_roi_filename if syn_roi_filename else None
    normalized_output = output_dir / normalized_filename if normalized_filename else None
    brain_mask_output = output_dir / brain_mask_filename if brain_mask_filename else None
    roi_brain_filtered_output = output_dir / roi_brain_filtered_filename if roi_brain_filtered_filename else None

    # Check if outputs exist and skip if requested
    if params.skip_if_exists and syn_output.exists():
        logger.info(f"Output exists, skipping registration: {syn_output}")
        return RegistrationResult(
            registered_image=syn_output,
            registered_roi=syn_roi_output if syn_roi_output and syn_roi_output.exists() else None,
            normalized_image=normalized_output if normalized_output and normalized_output.exists() else None,
            brain_mask=brain_mask_output if brain_mask_output and brain_mask_output.exists() else None,
            roi_mask_brain_filtered=roi_brain_filtered_output if roi_brain_filtered_output and roi_brain_filtered_output.exists() else None,
            rigid_image=rigid_output if rigid_output.exists() else None,
        )

    try:
        # Load images
        logger.info(f"Loading standard space: {standard_space_path}")
        fixed_image = ants.image_read(str(standard_space_path))
        
        logger.info(f"Loading moving image: {img_path}")
        moving_image = ants.image_read(str(img_path))

        # ====== STAGE 1: RIGID REGISTRATION ======
        logger.info("="*70)
        logger.info("STAGE 1: RIGID REGISTRATION")
        logger.info("="*70)
        
        tx_rigid = register_rigid(
            fixed_image=fixed_image,
            moving_image=moving_image,
            metric=params.rigid_metric,
            grad_step=params.rigid_grad_step,
            verbose=verbose
        )

        rigid_image = apply_rigid_transform(
            fixed_image=fixed_image,
            moving_image=moving_image,
            transforms=tx_rigid["fwdtransforms"]
        )
        ants.image_write(rigid_image, str(rigid_output))
        logger.info(f"✓ Rigid output: {rigid_output.name}")

        # Apply rigid transform to ROI if provided
        rigid_roi_image = None
        if roi_path and roi_path.exists():
            logger.info(f"Loading ROI: {roi_path}")
            roi_image = ants.image_read(str(roi_path))
            
            rigid_roi_image = apply_rigid_transform(
                fixed_image=fixed_image,
                moving_image=roi_image,
                transforms=tx_rigid["fwdtransforms"],
                interpolator="nearestNeighbor"
            )
            rigid_roi_output = output_dir / BIDSNamer.rigid_roi_mask(bids_config, roi_name)
            ants.image_write(rigid_roi_image, str(rigid_roi_output))
            logger.info(f"✓ Rigid ROI: {rigid_roi_output.name}")

        # ====== STAGE 2: SYN REGISTRATION ======
        logger.info("="*70)
        logger.info("STAGE 2: SYN (NONLINEAR) REGISTRATION")
        logger.info("="*70)
        
        tx_syn = register_syn(
            fixed_image=fixed_image,
            moving_image=rigid_image,
            grad_step=params.syn_grad_step,
            iterations=params.syn_iterations,
            verbose=verbose
        )

        syn_image = apply_rigid_transform(
            fixed_image=fixed_image,
            moving_image=rigid_image,
            transforms=tx_syn["fwdtransforms"]
        )
        ants.image_write(syn_image, str(syn_output))
        logger.info(f"✓ SyN output: {syn_output.name}")

        # Apply SyN transform to ROI if provided
        syn_roi_image = None
        if roi_path and rigid_roi_image is not None:
            syn_roi_image = apply_rigid_transform(
                fixed_image=fixed_image,
                moving_image=rigid_roi_image,
                transforms=tx_syn["fwdtransforms"],
                interpolator="nearestNeighbor"
            )
            ants.image_write(syn_roi_image, str(syn_roi_output))
            logger.info(f"✓ SyN ROI: {syn_roi_output.name}")

        # ====== POST-PROCESSING ======
        logger.info("="*70)
        logger.info("POST-PROCESSING")
        logger.info("="*70)

        syn_array = syn_image.numpy()
        
        # Normalize if requested
        if params.normalize and normalized_output:
            logger.info("Normalizing image intensities...")
            normalized_array = _normalize_image(syn_array)
            norm_image = ants.from_numpy(
                normalized_array,
                origin=syn_image.origin,
                spacing=syn_image.spacing,
                direction=syn_image.direction
            )
            ants.image_write(norm_image, str(normalized_output))
            logger.info(f"✓ Normalized: {normalized_output.name}")
        else:
            normalized_array = syn_array

        # Create brain mask if requested
        brain_mask = None
        if params.create_brain_mask and brain_mask_output:
            logger.info("Creating brain mask...")
            brain_mask = _create_brain_mask_from_normalized(
                normalized_array,
                threshold=params.brain_mask_threshold
            )
            brain_mask_img = ants.from_numpy(
                brain_mask.astype(np.float32),
                origin=syn_image.origin,
                spacing=syn_image.spacing,
                direction=syn_image.direction
            )
            ants.image_write(brain_mask_img, str(brain_mask_output))
            logger.info(f"✓ Brain mask: {brain_mask_output.name}")

        # Filter ROI by brain mask if requested
        if roi_path and syn_roi_image is not None and params.filter_roi_by_brain and brain_mask is not None:
            logger.info("Filtering ROI by brain mask...")
            syn_roi_array = syn_roi_image.numpy()
            roi_mask = (syn_roi_array > params.roi_mask_threshold).astype(np.uint8)
            roi_brain_filtered = roi_mask * brain_mask
            
            roi_brain_filtered_img = ants.from_numpy(
                roi_brain_filtered.astype(np.float32),
                origin=syn_roi_image.origin,
                spacing=syn_roi_image.spacing,
                direction=syn_roi_image.direction
            )
            ants.image_write(roi_brain_filtered_img, str(roi_brain_filtered_output))
            logger.info(f"✓ Brain-filtered ROI: {roi_brain_filtered_output.name}")

        logger.info("="*70)
        logger.info("✓ REGISTRATION COMPLETED SUCCESSFULLY")
        logger.info("="*70)

        return RegistrationResult(
            registered_image=syn_output,
            registered_roi=syn_roi_output,
            normalized_image=normalized_output,
            brain_mask=brain_mask_output,
            roi_mask_brain_filtered=roi_brain_filtered_output,
            rigid_image=rigid_output,
        )

    except Exception as e:
        logger.error(f"Registration failed: {e}", exc_info=True)
        raise


# Backward compatibility alias
def best_registration(
    img_path: Union[str, Path],
    standard_space_path: Union[str, Path],
    output_dir: Union[str, Path],
    roi_path: Optional[Union[str, Path]] = None,
    name: str = "FLAIR",
    params: Optional[RegistrationParams] = None,
    verbose: bool = False
) -> RegistrationResult:
    """
    Backward compatibility wrapper for register_to_standard_space.
    
    Converts old function signature to BIDS naming convention.
    """
    # Extract subject ID from output path if possible, else use generic naming
    subject_id = "unknown"
    session_id = None
    
    bids_config = BIDSNamingConfig(
        subject_id=subject_id,
        session_id=session_id,
        modality=name,
    )
    
    return register_to_standard_space(
        img_path=img_path,
        standard_space_path=standard_space_path,
        output_dir=output_dir,
        bids_config=bids_config,
        roi_path=roi_path,
        params=params,
        verbose=verbose
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("BIDS-conform Image Registration Module imported successfully")