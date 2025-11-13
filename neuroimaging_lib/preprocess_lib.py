#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized brain image processing utilities for neuroimaging pipelines.

Provides optimized functions for:
- N4 Bias Field Correction
- FSL-compliant NIfTI file operations
- Volume calculations
- Brain extraction
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Tuple, Union, Optional
from dataclasses import dataclass

import ants
import nibabel as nib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VolumeInfo:
    """Data class for volume information."""
    nonzero_voxels: float
    volume_mm3: float
    volume_ml: float


# ============================================================================
# N4 BIAS FIELD CORRECTION
# ============================================================================

def apply_bias_correction(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    shrink_factor: int = 4,
    convergence: Optional[dict] = None,
) -> Path:
    """
    Perform N4 Bias Field Correction using ANTsPy.
    
    Optimized with better error handling and validation.

    Parameters:
        input_path: Path to input NIfTI image.
        output_path: Path to save bias-corrected image.
        shrink_factor: Shrink factor for faster processing (default: 4).
        convergence: Convergence parameters dict. If None, uses defaults.

    Returns:
        Path: Path to the saved corrected image.

    Raises:
        FileNotFoundError: If input file doesn't exist.
        RuntimeError: If bias correction fails.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Validate input
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Skip if output already exists
    if output_path.exists():
        logger.info(f"Output already exists, skipping: {output_path}")
        return output_path

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Loading image: {input_path}")
        image = ants.image_read(str(input_path))

        logger.info("Applying N4 Bias Field Correction")
        corrected_image = ants.n4_bias_field_correction(
            image,
            shrink_factor=shrink_factor,
            convergence=convergence
        )

        logger.info(f"Saving corrected image: {output_path}")
        ants.image_write(corrected_image, str(output_path))

        return output_path

    except AttributeError as ae:
        logger.error(f"ANTsPy attribute error: {ae}", exc_info=True)
        raise RuntimeError(f"ANTsPy error in bias correction: {ae}") from ae
    except Exception as e:
        logger.error(f"Error in bias correction: {e}", exc_info=True)
        raise RuntimeError(f"Bias correction failed: {e}") from e


# ============================================================================
# NIfTI FILE VALIDATION
# ============================================================================

def _validate_nifti(
    file_path: Union[str, Path],
    verbose: bool = False
) -> bool:
    """
    Validate NIfTI file integrity.

    Parameters:
        file_path: Path to NIfTI file to validate.
        verbose: If True, log validation results.

    Returns:
        bool: True if file is valid NIfTI, False otherwise.
    """
    file_path = Path(file_path)
    
    try:
        # Check file exists
        if not file_path.exists():
            if verbose:
                logger.warning(f"File not found: {file_path}")
            return False

        # Try to load with nibabel
        nib.load(str(file_path))
        if verbose:
            logger.info(f"✓ NIfTI valid: {file_path}")
        return True

    except Exception as e:
        if verbose:
            logger.warning(f"✗ NIfTI invalid: {file_path} - {e}")
        return False


# ============================================================================
# FSL FILE OPERATIONS
# ============================================================================

def fsl_copy(
    src: Union[str, Path],
    dst: Union[str, Path],
    label: str = "",
    force: bool = False,
    use_fallback: bool = True,
) -> bool:
    """
    Copy NIfTI file FSL-compliant and validate result.
    
    Optimized: Uses subprocess lists instead of strings, better error handling,
    professional logging instead of print statements.

    Parameters:
        src: Source NIfTI file path.
        dst: Destination NIfTI file path.
        label: Optional label for logging (max 10 chars).
        force: If True, overwrite existing destination.
        use_fallback: If True, fallback to shutil.copy2 if fslmaths fails.

    Returns:
        bool: True if destination exists and validates, False otherwise.
    """
    import shutil

    src = Path(src)
    dst = Path(dst)
    label = label[:10] if label else ""

    # Validate source
    if not src.exists():
        logger.error(f"[{label}] Source not found: {src}")
        return False

    # Handle existing destination
    if dst.exists():
        if not force:
            logger.info(f"[{label}] Destination exists, skipping: {dst}")
            return _validate_nifti(dst, verbose=True)
        else:
            logger.warning(f"[{label}] Overwriting existing destination: {dst}")
            try:
                dst.unlink()
            except OSError as e:
                logger.warning(f"[{label}] Failed to remove old file: {e}")

    # Create output directory
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Strategy 1: Use fslmaths (preserves datatype, header orientation)
    try:
        cmd = ["fslmaths", str(src), "-mul", "1", str(dst)]
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=300
        )
        logger.info(f"[{label}] ✓ Copied via fslmaths: {dst}")

        # Validate fslmaths result
        if _validate_nifti(dst, verbose=True):
            return True
        else:
            logger.warning(f"[{label}] fslmaths copy corrupted, trying fallback")

    except FileNotFoundError:
        logger.warning(f"[{label}] fslmaths not found, using fallback")
    except subprocess.CalledProcessError as e:
        logger.warning(f"[{label}] fslmaths failed: {e.stderr}")
    except subprocess.TimeoutExpired:
        logger.warning(f"[{label}] fslmaths timed out")
    except Exception as e:
        logger.warning(f"[{label}] fslmaths error: {e}")

    # Strategy 2: Fallback to shutil.copy2
    if use_fallback:
        try:
            shutil.copy2(src, dst)
            logger.info(f"[{label}] ✓ Copied via shutil: {dst}")
            return _validate_nifti(dst, verbose=True)

        except Exception as e:
            logger.error(f"[{label}] ✗ Copy failed: {e}")
            return False

    logger.error(f"[{label}] All copy strategies failed")
    return False


# ============================================================================
# VOLUME CALCULATIONS
# ============================================================================

def get_volume(
    image_path: Union[str, Path],
    verbose: bool = False
) -> VolumeInfo:
    """
    Calculate volume from brain-extracted image using fslstats.
    
    Optimized: Uses subprocess list (secure), better error handling.

    Parameters:
        image_path: Path to brain-extracted image.
        verbose: If True, log details.

    Returns:
        VolumeInfo: Dataclass containing volume information.

    Raises:
        FileNotFoundError: If image file doesn't exist.
        RuntimeError: If fslstats fails.
    """
    image_path = Path(image_path)

    # Validate input
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        if verbose:
            logger.info(f"Calculating volume for: {image_path}")

        # Use subprocess list (more secure than shell=True)
        cmd = ["fslstats", str(image_path), "-V"]
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=60
        )

        # Parse output
        try:
            nonzero_voxels, volume_mm3 = map(float, result.stdout.split())
        except ValueError as e:
            raise RuntimeError(
                f"Failed to parse fslstats output: {result.stdout}"
            ) from e

        volume_ml = volume_mm3 / 1000  # Convert mm³ to ml

        if verbose:
            logger.info(
                f"Volume: {nonzero_voxels:.0f} voxels, "
                f"{volume_mm3:.2f} mm³, {volume_ml:.2f} ml"
            )

        return VolumeInfo(
            nonzero_voxels=nonzero_voxels,
            volume_mm3=volume_mm3,
            volume_ml=volume_ml
        )

    except FileNotFoundError:
        raise RuntimeError("fslstats not found. Make sure FSL is installed.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"fslstats error: {e.stderr}") from e
    except subprocess.TimeoutExpired:
        raise RuntimeError("fslstats timed out") from None
    except Exception as e:
        raise RuntimeError(f"Volume calculation failed: {e}") from e


# ============================================================================
# BRAIN EXTRACTION
# ============================================================================

def brain_extraction_hdbet(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    output_mask: Optional[Union[str, Path]] = None,
    device: str = "cpu",
    tta: bool = True,
    verbose: bool = False
) -> Path:
    """
    Perform brain extraction using HD-BET.
    
    Optimized version of hdbet function with proper error handling.

    Parameters:
        input_path: Path to input MRI image.
        output_path: Path to save brain-extracted image.
        output_mask: Optional path to save brain mask.
        device: Device to use ("cpu" or "cuda").
        tta: If True, use test-time augmentation (more accurate, slower).
        verbose: If True, log details.

    Returns:
        Path: Path to brain-extracted image.

    Raises:
        FileNotFoundError: If input file or hd-bet not found.
        RuntimeError: If brain extraction fails.

    Note:
        Please cite HD-BET if you use this function:
        Isensee F, Schell M, et al. Automated brain extraction of multi-sequence MRI
        using artificial neural networks. arXiv preprint arXiv:1901.11341, 2019.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Validate input
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if verbose:
            logger.info(f"Starting brain extraction: {input_path}")

        # Build hd-bet command
        cmd = [
            "hd-bet",
            "-i", str(input_path),
            "-o", str(output_path),
            "-device", device,
        ]

        # Add optional arguments
        if not tta:
            cmd.append("--disable_tta")

        if output_mask:
            cmd.append("--save_bet_mask")

        if verbose:
            logger.info(f"Running command: {' '.join(cmd)}")

        # Run hd-bet
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=600
        )

        if verbose:
            logger.info(f"Brain extraction completed: {output_path}")
            if result.stdout:
                logger.debug(f"hd-bet output: {result.stdout}")

        # Validate output
        if not output_path.exists():
            raise RuntimeError(f"Brain-extracted image not created: {output_path}")

        if not _validate_nifti(output_path, verbose=verbose):
            raise RuntimeError(f"Brain-extracted image is invalid: {output_path}")

        return output_path

    except FileNotFoundError:
        raise RuntimeError(
            "hd-bet not found. Install with: pip install hd-bet"
        ) from None
    except subprocess.CalledProcessError as e:
        logger.error(f"hd-bet error: {e.stderr}")
        raise RuntimeError(f"Brain extraction failed: {e.stderr}") from e
    except subprocess.TimeoutExpired:
        raise RuntimeError("Brain extraction timed out (10 minutes)") from None
    except Exception as e:
        logger.error(f"Brain extraction error: {e}", exc_info=True)
        raise RuntimeError(f"Brain extraction failed: {e}") from e


def brain_extraction_bet(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    fractional_intensity: float = 0.5,
    vertical_gradient: float = 0.0,
    verbose: bool = False
) -> Path:
    """
    Perform brain extraction using FSL's BET.
    
    Lighter-weight alternative to HD-BET.

    Parameters:
        input_path: Path to input MRI image.
        output_path: Path to save brain-extracted image.
        fractional_intensity: Fractional intensity threshold (0-1).
        vertical_gradient: Vertical gradient in fractional intensity (-1 to 1).
        verbose: If True, log details.

    Returns:
        Path: Path to brain-extracted image.

    Raises:
        FileNotFoundError: If input file or BET not found.
        RuntimeError: If brain extraction fails.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Validate input
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if verbose:
            logger.info(f"Starting BET brain extraction: {input_path}")

        # Build BET command
        cmd = [
            "bet",
            str(input_path),
            str(output_path),
            "-f", str(fractional_intensity),
            "-g", str(vertical_gradient),
        ]

        if verbose:
            logger.info(f"Running command: {' '.join(cmd)}")

        # Run BET
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=300
        )

        if verbose:
            logger.info(f"BET brain extraction completed: {output_path}")

        # Validate output
        if not output_path.exists():
            raise RuntimeError(f"Brain-extracted image not created: {output_path}")

        if not _validate_nifti(output_path, verbose=verbose):
            raise RuntimeError(f"Brain-extracted image is invalid: {output_path}")

        return output_path

    except FileNotFoundError:
        raise RuntimeError(
            "bet not found. Make sure FSL is installed."
        ) from None
    except subprocess.CalledProcessError as e:
        logger.error(f"BET error: {e.stderr}")
        raise RuntimeError(f"Brain extraction failed: {e.stderr}") from e
    except subprocess.TimeoutExpired:
        raise RuntimeError("Brain extraction timed out (5 minutes)") from None
    except Exception as e:
        logger.error(f"Brain extraction error: {e}", exc_info=True)
        raise RuntimeError(f"Brain extraction failed: {e}") from e


if __name__ == "__main__":
    # Example usage
    logger.info("Brain Processing Library imported successfully")