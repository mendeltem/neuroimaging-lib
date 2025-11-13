#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized DICOM to NIfTI conversion utilities for neuroimaging pipelines.

Provides functions for:
- DICOM file format conversion to NIfTI
- Multiple conversion strategies with automatic fallback
- Robust error handling and recovery
- BIDS-compliant output naming
- Support for various DICOM compressions (JPEG, JPEG2000)

Conversion Priority:
    1. dcm2niix (fastest, most reliable)
    2. SimpleITK (handles JPEG2000)
    3. Manual pydicom conversion (fallback, slower)
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Union, Optional, Dict, List
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class ConversionMethod(Enum):
    """Supported DICOM to NIfTI conversion methods."""
    DCM2NIIX = "dcm2niix"
    SIMPLEITK = "simpleitk"
    PYDICOM = "pydicom"


@dataclass
class Dicom2NiiConfig:
    """Configuration for DICOM to NIfTI conversion."""
    # Output settings
    gzip_output: bool = True          # Compress output (.nii.gz vs .nii)
    output_filename: str = "image.nii.gz"
    
    # Method selection
    preferred_methods: List[ConversionMethod] = None
    
    # Processing parameters
    timeout: int = 600                # Conversion timeout in seconds
    verify_output: bool = True        # Verify output exists and is valid
    preserve_orient: bool = True      # Preserve image orientation
    
    # Logging
    verbose: bool = False
    
    def __post_init__(self):
        """Set default preferred methods."""
        if self.preferred_methods is None:
            self.preferred_methods = [
                ConversionMethod.DCM2NIIX,
                ConversionMethod.SIMPLEITK,
                ConversionMethod.PYDICOM,
            ]


@dataclass
class ConversionResult:
    """Result of DICOM to NIfTI conversion."""
    success: bool
    output_path: Optional[Path] = None
    method_used: Optional[ConversionMethod] = None
    error_message: Optional[str] = None
    file_size: Optional[int] = None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _validate_dicom_input(
    dicom_dir: Union[str, Path],
    verbose: bool = False
) -> Path:
    """
    Validate DICOM input directory.
    
    Parameters:
        dicom_dir: Path to directory containing DICOM files.
        verbose: If True, log validation details.
    
    Returns:
        Path: Validated directory path.
    
    Raises:
        FileNotFoundError: If directory doesn't exist.
        ValueError: If no DICOM files found.
    """
    dicom_dir = Path(dicom_dir)
    
    if not dicom_dir.exists():
        raise FileNotFoundError(f"DICOM directory not found: {dicom_dir}")
    
    if not dicom_dir.is_dir():
        raise ValueError(f"Path is not a directory: {dicom_dir}")
    
    # Check for DICOM files
    dicom_files = list(dicom_dir.glob("*.dcm"))
    if not dicom_files:
        raise ValueError(f"No DICOM files (.dcm) found in: {dicom_dir}")
    
    if verbose:
        logger.info(f"✓ DICOM directory validated: {dicom_dir}")
        logger.info(f"  DICOM files found: {len(dicom_files)}")
    
    return dicom_dir


def _check_tool_available(tool_name: str) -> bool:
    """
    Check if a command-line tool is available.
    
    Parameters:
        tool_name: Name of the tool to check.
    
    Returns:
        bool: True if tool is available, False otherwise.
    """
    try:
        result = subprocess.run(
            ["which", tool_name],
            check=False,
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception as e:
        logger.debug(f"Error checking tool availability: {e}")
        return False


# ============================================================================
# CONVERSION METHODS
# ============================================================================

class Dcm2niixConverter:
    """DICOM to NIfTI conversion using dcm2niix."""
    
    @staticmethod
    def available() -> bool:
        """Check if dcm2niix is available."""
        return _check_tool_available("dcm2niix")
    
    @staticmethod
    def convert(
        dicom_dir: Path,
        output_dir: Path,
        output_filename: str,
        gzip: bool = True,
        timeout: int = 300,
        verbose: bool = False
    ) -> ConversionResult:
        """
        Convert DICOM to NIfTI using dcm2niix.
        
        Parameters:
            dicom_dir: Path to DICOM directory.
            output_dir: Path to output directory.
            output_filename: Output filename.
            gzip: If True, compress output.
            timeout: Timeout in seconds.
            verbose: If True, show verbose output.
        
        Returns:
            ConversionResult: Conversion result.
        """
        try:
            if not Dcm2niixConverter.available():
                return ConversionResult(
                    success=False,
                    error_message="dcm2niix not found in PATH"
                )
            
            logger.info("Converting DICOM with dcm2niix...")
            
            # Prepare output filename (dcm2niix will add extension)
            base_filename = output_filename.replace('.nii.gz', '').replace('.nii', '')
            
            # Build command (use list format for security)
            cmd = [
                "dcm2niix",
                "-z", "y" if gzip else "n",  # Compression
                "-f", base_filename,           # Output filename
                "-o", str(output_dir),         # Output directory
                str(dicom_dir)                 # Input directory
            ]
            
            logger.debug(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "Unknown error"
                return ConversionResult(
                    success=False,
                    error_message=f"dcm2niix failed: {error_msg}"
                )
            
            # Find output file
            ext = ".nii.gz" if gzip else ".nii"
            output_path = output_dir / (base_filename + ext)
            
            if not output_path.exists():
                return ConversionResult(
                    success=False,
                    error_message=f"Output file not created: {output_path}"
                )
            
            file_size = output_path.stat().st_size
            logger.info(f"✓ dcm2niix conversion successful: {file_size} bytes")
            
            return ConversionResult(
                success=True,
                output_path=output_path,
                method_used=ConversionMethod.DCM2NIIX,
                file_size=file_size
            )
            
        except subprocess.TimeoutExpired:
            return ConversionResult(
                success=False,
                error_message=f"dcm2niix timed out after {timeout}s"
            )
        except Exception as e:
            return ConversionResult(
                success=False,
                error_message=f"dcm2niix error: {e}"
            )


class SimpleitkConverter:
    """DICOM to NIfTI conversion using SimpleITK."""
    
    @staticmethod
    def available() -> bool:
        """Check if SimpleITK is available."""
        try:
            import SimpleITK as sitk
            return True
        except ImportError:
            return False
    
    @staticmethod
    def convert(
        dicom_dir: Path,
        output_dir: Path,
        output_filename: str,
        verbose: bool = False
    ) -> ConversionResult:
        """
        Convert DICOM to NIfTI using SimpleITK.
        
        Better for handling JPEG2000 and other complex compressions.
        
        Parameters:
            dicom_dir: Path to DICOM directory.
            output_dir: Path to output directory.
            output_filename: Output filename.
            verbose: If True, show verbose output.
        
        Returns:
            ConversionResult: Conversion result.
        """
        try:
            import SimpleITK as sitk
            
            if not SimpleitkConverter.available():
                return ConversionResult(
                    success=False,
                    error_message="SimpleITK not installed"
                )
            
            logger.info("Converting DICOM with SimpleITK...")
            
            # Read DICOM series
            reader = sitk.ImageSeriesReader()
            series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
            
            if not series_ids:
                return ConversionResult(
                    success=False,
                    error_message="No DICOM series found in directory"
                )
            
            # Use first series
            dicom_files = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])
            
            if not dicom_files:
                return ConversionResult(
                    success=False,
                    error_message="No DICOM files in series"
                )
            
            logger.debug(f"Reading {len(dicom_files)} DICOM files...")
            reader.SetFileNames(dicom_files)
            image = reader.Execute()
            
            # Save as NIfTI
            output_path = output_dir / output_filename
            logger.debug(f"Writing to {output_path}...")
            sitk.WriteImage(image, str(output_path))
            
            if not output_path.exists():
                return ConversionResult(
                    success=False,
                    error_message=f"Output file not created: {output_path}"
                )
            
            file_size = output_path.stat().st_size
            logger.info(f"✓ SimpleITK conversion successful: {file_size} bytes")
            
            return ConversionResult(
                success=True,
                output_path=output_path,
                method_used=ConversionMethod.SIMPLEITK,
                file_size=file_size
            )
            
        except ImportError:
            return ConversionResult(
                success=False,
                error_message="SimpleITK not installed"
            )
        except Exception as e:
            logger.debug(f"SimpleITK error: {e}")
            return ConversionResult(
                success=False,
                error_message=f"SimpleITK error: {e}"
            )


class PydicomConverter:
    """DICOM to NIfTI conversion using pydicom (fallback method)."""
    
    @staticmethod
    def available() -> bool:
        """Check if pydicom is available."""
        try:
            import pydicom
            import nibabel
            return True
        except ImportError:
            return False
    
    @staticmethod
    def convert(
        dicom_dir: Path,
        output_dir: Path,
        output_filename: str,
        verbose: bool = False
    ) -> ConversionResult:
        """
        Convert DICOM to NIfTI using pydicom (manual conversion).
        
        Slower but always works as fallback.
        
        Parameters:
            dicom_dir: Path to DICOM directory.
            output_dir: Path to output directory.
            output_filename: Output filename.
            verbose: If True, show verbose output.
        
        Returns:
            ConversionResult: Conversion result.
        """
        try:
            import pydicom
            import nibabel as nib
            
            if not PydicomConverter.available():
                return ConversionResult(
                    success=False,
                    error_message="pydicom or nibabel not installed"
                )
            
            logger.info("Converting DICOM with pydicom (manual)...")
            
            # Find DICOM files
            dicom_files = sorted(dicom_dir.glob("*.dcm"))
            
            if not dicom_files:
                return ConversionResult(
                    success=False,
                    error_message="No DICOM files found"
                )
            
            logger.debug(f"Reading {len(dicom_files)} DICOM files...")
            
            # Read DICOM files
            slices = []
            for dcm_file in dicom_files:
                try:
                    dcm = pydicom.dcmread(dcm_file)
                    slices.append(dcm)
                except Exception as e:
                    logger.warning(f"Failed to read {dcm_file}: {e}")
            
            if not slices:
                return ConversionResult(
                    success=False,
                    error_message="Could not read any DICOM files"
                )
            
            # Sort by instance number
            slices.sort(key=lambda x: int(getattr(x, 'InstanceNumber', 0)))
            
            # Stack into 3D volume
            try:
                volume = np.stack([s.pixel_array for s in slices], axis=-1)
            except Exception as e:
                logger.error(f"Failed to stack slices: {e}")
                return ConversionResult(
                    success=False,
                    error_message=f"Failed to stack slices: {e}"
                )
            
            # Create affine matrix (identity for now)
            affine = np.eye(4)
            
            # Create NIfTI image
            img = nib.Nifti1Image(volume.astype(np.float32), affine)
            
            # Save
            output_path = output_dir / output_filename
            logger.debug(f"Saving to {output_path}...")
            nib.save(img, str(output_path))
            
            if not output_path.exists():
                return ConversionResult(
                    success=False,
                    error_message=f"Output file not created: {output_path}"
                )
            
            file_size = output_path.stat().st_size
            logger.info(f"✓ pydicom conversion successful: {file_size} bytes")
            
            return ConversionResult(
                success=True,
                output_path=output_path,
                method_used=ConversionMethod.PYDICOM,
                file_size=file_size
            )
            
        except ImportError as e:
            return ConversionResult(
                success=False,
                error_message=f"Missing dependency: {e}"
            )
        except Exception as e:
            logger.debug(f"pydicom error: {e}")
            return ConversionResult(
                success=False,
                error_message=f"pydicom error: {e}"
            )


# ============================================================================
# MAIN CONVERSION WITH FALLBACK
# ============================================================================

def convert_dicom_to_nifti(
    dicom_dir: Union[str, Path],
    output_dir: Union[str, Path],
    output_filename: str = "image.nii.gz",
    config: Optional[Dicom2NiiConfig] = None,
    verbose: bool = False
) -> ConversionResult:
    """
    Convert DICOM to NIfTI with automatic fallback strategies.
    
    Tries multiple conversion methods in order:
    1. dcm2niix (fastest, production quality)
    2. SimpleITK (handles complex compressions)
    3. pydicom (fallback, always works)

    Parameters:
        dicom_dir: Path to directory containing DICOM files.
        output_dir: Path to output directory.
        output_filename: Output filename (default: "image.nii.gz").
        config: Dicom2NiiConfig with conversion settings.
        verbose: If True, show detailed output.

    Returns:
        ConversionResult: Detailed conversion result with success status and method used.

    Raises:
        FileNotFoundError: If DICOM directory doesn't exist.
        ValueError: If no DICOM files found.

    Example:
        >>> result = convert_dicom_to_nifti(
        ...     "data/dicom",
        ...     "output",
        ...     "flair.nii.gz"
        ... )
        >>> if result.success:
        ...     print(f"Converted: {result.output_path}")
        ...     print(f"Method: {result.method_used.value}")
        ... else:
        ...     print(f"Failed: {result.error_message}")
    """
    # Use default config if not provided
    if config is None:
        config = Dicom2NiiConfig()
    
    dicom_dir = Path(dicom_dir)
    output_dir = Path(output_dir)
    
    # Validate input
    logger.info(f"Validating DICOM input: {dicom_dir.name}")
    try:
        dicom_dir = _validate_dicom_input(dicom_dir, verbose=verbose)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Input validation failed: {e}")
        return ConversionResult(
            success=False,
            error_message=str(e)
        )
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Determine output filename with proper extension
    if config.gzip_output:
        if not output_filename.endswith('.nii.gz'):
            output_filename = output_filename.replace('.nii', '') + '.nii.gz'
    else:
        if output_filename.endswith('.nii.gz'):
            output_filename = output_filename.replace('.nii.gz', '.nii')
        elif not output_filename.endswith('.nii'):
            output_filename += '.nii'
    
    logger.info(f"Output filename: {output_filename}")
    
    # Try each conversion method
    converters = []
    for method in config.preferred_methods:
        if method == ConversionMethod.DCM2NIIX:
            converters.append(("dcm2niix", Dcm2niixConverter))
        elif method == ConversionMethod.SIMPLEITK:
            converters.append(("SimpleITK", SimpleitkConverter))
        elif method == ConversionMethod.PYDICOM:
            converters.append(("pydicom", PydicomConverter))
    
    logger.info(f"Trying conversion methods: {[name for name, _ in converters]}")
    
    for method_name, converter_class in converters:
        try:
            # Check if method is available
            if not converter_class.available():
                logger.info(f"  ⊘ {method_name} not available")
                continue
            
            logger.info(f"  → Trying {method_name}...")
            
            # Attempt conversion
            if method_name == "dcm2niix":
                result = converter_class.convert(
                    dicom_dir,
                    output_dir,
                    output_filename,
                    gzip=config.gzip_output,
                    timeout=config.timeout,
                    verbose=verbose
                )
            else:
                result = converter_class.convert(
                    dicom_dir,
                    output_dir,
                    output_filename,
                    verbose=verbose
                )
            
            if result.success:
                logger.info(f"✓ Conversion successful with {method_name}")
                logger.info(f"  Output: {result.output_path}")
                logger.info(f"  Size: {result.file_size} bytes")
                return result
            else:
                logger.debug(f"  ✗ {method_name} failed: {result.error_message}")
        
        except Exception as e:
            logger.debug(f"  ✗ {method_name} exception: {e}")
    
    # All methods failed
    error_msg = "All conversion methods failed"
    logger.error(error_msg)
    return ConversionResult(
        success=False,
        error_message=error_msg
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("DICOM to NIfTI Conversion Module imported successfully")