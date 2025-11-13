#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized VOI to NIfTI conversion utilities for neuroimaging pipelines.

Provides functions for:
- VOI file format conversion to NIfTI
- MATLAB-based conversion (requires MATLAB)
- BIDS-compliant output naming
- Batch processing support

VOI Format: Brainvisa/Anatomist region of interest format
Output: NIfTI (.nii.gz)
"""

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Union, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VOI2NiiConfig:
    """Configuration for VOI to NIfTI conversion."""
    matlab_path: Optional[Path] = None  # Additional MATLAB path to add
    timeout: int = 300                   # Conversion timeout in seconds
    cleanup_temp: bool = True           # Clean up temporary files
    verify_output: bool = True          # Verify output file exists
    bids_compliant: bool = False        # Use BIDS naming for output


class MATLABManager:
    """Manage MATLAB availability and execution."""
    
    @staticmethod
    def check_matlab_available() -> bool:
        """
        Check if MATLAB is available in PATH.
        
        Returns:
            bool: True if MATLAB is available, False otherwise.
        """
        try:
            result = subprocess.run(
                ["which", "matlab"],
                check=False,
                capture_output=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception as e:
            logger.warning(f"Error checking MATLAB availability: {e}")
            return False
    
    @staticmethod
    def get_matlab_version() -> Optional[str]:
        """
        Get MATLAB version.
        
        Returns:
            Optional[str]: MATLAB version string, or None if not available.
        """
        try:
            result = subprocess.run(
                ["matlab", "-batch", "version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                # Parse version from output
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'MATLAB' in line and 'Version' in line:
                        return line.strip()
            return None
        except Exception as e:
            logger.warning(f"Error getting MATLAB version: {e}")
            return None
    
    @staticmethod
    def run_matlab_script(
        script_content: str,
        timeout: int = 300,
        verbose: bool = False
    ) -> tuple[int, str, str]:
        """
        Run MATLAB script safely without shell injection.
        
        Parameters:
            script_content: MATLAB script code to execute.
            timeout: Timeout in seconds.
            verbose: If True, show MATLAB output.
        
        Returns:
            Tuple[int, str, str]: (return_code, stdout, stderr)
        
        Raises:
            RuntimeError: If MATLAB execution fails.
            subprocess.TimeoutExpired: If MATLAB times out.
        """
        # Create temporary MATLAB script
        with tempfile.NamedTemporaryFile(
            mode="w",
            delete=False,
            suffix=".m",
            encoding="utf-8"
        ) as f:
            f.write(script_content)
            script_path = f.name
        
        try:
            logger.debug(f"MATLAB script: {script_path}")
            
            # Use list format (no shell=True for security)
            cmd = [
                "matlab",
                "-batch",
                f"run('{script_path}');"
            ]
            
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if verbose and result.stdout:
                logger.info(f"MATLAB output:\n{result.stdout}")
            
            if result.stderr:
                logger.warning(f"MATLAB stderr:\n{result.stderr}")
            
            return result.returncode, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired as e:
            logger.error(f"MATLAB execution timed out after {timeout} seconds")
            raise
        finally:
            # Clean up temporary script
            try:
                os.remove(script_path)
                logger.debug(f"Removed temporary script: {script_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary script: {e}")


# ============================================================================
# VOI VALIDATION AND UTILITIES
# ============================================================================

def _validate_voi_input(
    input_path: Union[str, Path],
    verbose: bool = False
) -> Path:
    """
    Validate VOI input file.
    
    Parameters:
        input_path: Path to VOI file.
        verbose: If True, log validation details.
    
    Returns:
        Path: Validated input path as Path object.
    
    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file is not a VOI file.
    """
    input_path = Path(input_path)
    
    # Check file exists
    if not input_path.exists():
        raise FileNotFoundError(f"VOI file not found: {input_path}")
    
    # Check file has .voi extension
    if input_path.suffix.lower() != '.voi':
        raise ValueError(
            f"File must have .voi extension: {input_path.name} "
            f"(has {input_path.suffix})"
        )
    
    # Check file is not empty
    if input_path.stat().st_size == 0:
        raise ValueError(f"VOI file is empty: {input_path}")
    
    if verbose:
        logger.info(f"✓ VOI file validated: {input_path}")
        logger.info(f"  Size: {input_path.stat().st_size} bytes")
    
    return input_path


def _create_matlab_conversion_script(
    input_file: str,
    input_dir: Union[str, Path],
    matlab_path: Optional[Union[str, Path]] = None
) -> str:
    """
    Create MATLAB script for VOI to NIfTI conversion.
    
    Parameters:
        input_file: VOI filename (without path).
        input_dir: Directory containing VOI file.
        matlab_path: Optional additional MATLAB path to add.
    
    Returns:
        str: MATLAB script code.
    """
    # Escape paths for MATLAB
    input_dir_str = str(input_dir).replace("\\", "\\\\")
    
    script = f"% VOI to NIfTI conversion script\n"
    script += f"% Generated automatically\n\n"
    
    # Add custom MATLAB path if provided
    if matlab_path:
        matlab_path_str = str(matlab_path).replace("\\", "\\\\")
        script += f"addpath('{matlab_path_str}');\n"
    
    # Change to input directory
    script += f"cd('{input_dir_str}');\n\n"
    
    # Run conversion
    script += f"% Load VOI and convert to NIfTI\n"
    script += f"try\n"
    script += f"    voi2nii('{input_file}');\n"
    script += f"    disp('✓ VOI conversion successful');\n"
    script += f"catch ME\n"
    script += f"    disp(['✗ VOI conversion failed: ' ME.message]);\n"
    script += f"    exit(1);\n"
    script += f"end\n\n"
    
    script += f"exit(0);\n"
    
    return script


def _determine_output_filename(
    input_path: Path,
    output_path: Optional[Path] = None
) -> Path:
    """
    Determine output filename after VOI conversion.
    
    MATLAB's voi2nii function creates output with same base name + .nii
    
    Parameters:
        input_path: Original VOI file path.
        output_path: Optional explicit output path.
    
    Returns:
        Path: Output NIfTI file path.
    """
    if output_path:
        return Path(output_path)
    
    # Default: same directory, replace .voi with .nii
    output_filename = input_path.stem + ".nii"
    output_path = input_path.parent / output_filename
    
    return output_path


# ============================================================================
# VOI TO NIFTI CONVERSION
# ============================================================================

def voi2nii(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    config: Optional[VOI2NiiConfig] = None,
    verbose: bool = False
) -> Path:
    """
    Convert VOI (Brainvisa ROI) file to NIfTI format using MATLAB.
    
    Optimized with:
    - Security improvements (no shell injection)
    - Professional error handling
    - Type hints
    - Proper cleanup
    - MATLAB validation
    - Comprehensive logging

    Parameters:
        input_path: Path to VOI file (.voi).
        output_path: Optional explicit output path. If None, uses input directory.
        config: VOI2NiiConfig with conversion settings. If None, uses defaults.
        verbose: If True, show detailed conversion output.

    Returns:
        Path: Path to converted NIfTI file.

    Raises:
        FileNotFoundError: If VOI file doesn't exist.
        ValueError: If input is not a .voi file.
        RuntimeError: If MATLAB is not available or conversion fails.

    Example:
        >>> nifti_path = voi2nii("region.voi", "output/region.nii")
        >>> print(nifti_path)
        PosixPath('output/region.nii')
    """
    # Use default config if not provided
    if config is None:
        config = VOI2NiiConfig()
    
    input_path = Path(input_path)
    
    # Validate input
    logger.info(f"Validating VOI input: {input_path.name}")
    input_path = _validate_voi_input(input_path, verbose=verbose)
    
    # Check MATLAB availability
    logger.info("Checking MATLAB availability...")
    if not MATLABManager.check_matlab_available():
        raise RuntimeError(
            "MATLAB not found in PATH. "
            "Please ensure MATLAB is installed and accessible."
        )
    
    matlab_version = MATLABManager.get_matlab_version()
    if matlab_version:
        logger.info(f"Using {matlab_version}")
    
    # Determine output path
    output_path = _determine_output_filename(input_path, output_path)
    output_path = Path(output_path)
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output will be saved to: {output_path}")
    
    # Validate config
    if config.timeout <= 0:
        raise ValueError(f"Timeout must be positive: {config.timeout}")
    
    try:
        # Create MATLAB conversion script
        logger.info("Creating MATLAB conversion script...")
        script = _create_matlab_conversion_script(
            input_path.name,
            input_path.parent,
            config.matlab_path
        )
        
        if verbose:
            logger.debug(f"MATLAB script:\n{script}")
        
        # Run MATLAB conversion
        logger.info(f"Running VOI to NIfTI conversion (timeout: {config.timeout}s)...")
        return_code, stdout, stderr = MATLABManager.run_matlab_script(
            script,
            timeout=config.timeout,
            verbose=verbose
        )
        
        # Check for MATLAB errors
        if return_code != 0:
            error_msg = f"MATLAB conversion failed (code {return_code})"
            if stderr:
                error_msg += f"\nError output:\n{stderr}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Find converted file
        logger.info("Locating converted NIfTI file...")
        temp_nifti = input_path.parent / (input_path.stem + ".nii")
        
        if not temp_nifti.exists():
            raise RuntimeError(
                f"MATLAB conversion completed but output not found: {temp_nifti}"
            )
        
        logger.info(f"✓ Conversion successful: {temp_nifti}")
        
        # Verify output if requested
        if config.verify_output:
            file_size = temp_nifti.stat().st_size
            if file_size == 0:
                raise RuntimeError(
                    f"Output file is empty: {temp_nifti}"
                )
            logger.info(f"Output file verified: {file_size} bytes")
        
        # Move to final destination if different
        if temp_nifti != output_path:
            logger.info(f"Moving to final destination: {output_path}")
            shutil.move(str(temp_nifti), str(output_path))
        
        logger.info(f"✓ VOI conversion completed: {output_path.name}")
        return output_path
        
    except subprocess.TimeoutExpired:
        logger.error(f"VOI conversion timed out after {config.timeout} seconds")
        raise RuntimeError(
            f"MATLAB conversion timed out after {config.timeout} seconds"
        ) from None
    except Exception as e:
        logger.error(f"VOI conversion failed: {e}", exc_info=True)
        raise
    finally:
        # Cleanup temporary files if requested
        if config.cleanup_temp:
            temp_nifti = input_path.parent / (input_path.stem + ".nii")
            if temp_nifti.exists() and temp_nifti != output_path:
                try:
                    temp_nifti.unlink()
                    logger.debug(f"Cleaned up temporary file: {temp_nifti}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup temporary file: {e}")


# ============================================================================
# BATCH CONVERSION
# ============================================================================

def batch_voi2nii(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    config: Optional[VOI2NiiConfig] = None,
    pattern: str = "*.voi",
    recursive: bool = True,
    verbose: bool = False
) -> dict:
    """
    Batch convert multiple VOI files to NIfTI.
    
    Parameters:
        input_dir: Directory containing VOI files.
        output_dir: Directory to save NIfTI files.
        config: VOI2NiiConfig with conversion settings.
        pattern: File pattern to match (default: "*.voi").
        recursive: If True, search subdirectories recursively.
        verbose: If True, show detailed output.
    
    Returns:
        dict: Results dictionary with:
            - 'successful': List of successfully converted files
            - 'failed': List of files that failed
            - 'total': Total files processed
    
    Example:
        >>> results = batch_voi2nii("data/voi", "data/nifti")
        >>> print(f"Converted {len(results['successful'])} files")
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Find VOI files
    logger.info(f"Searching for VOI files in {input_dir}")
    if recursive:
        voi_files = list(input_dir.rglob(pattern))
    else:
        voi_files = list(input_dir.glob(pattern))
    
    if not voi_files:
        logger.warning(f"No VOI files found matching pattern: {pattern}")
        return {'successful': [], 'failed': [], 'total': 0}
    
    logger.info(f"Found {len(voi_files)} VOI files to convert")
    
    results = {'successful': [], 'failed': [], 'total': len(voi_files)}
    
    # Convert each file
    for idx, voi_file in enumerate(voi_files, 1):
        try:
            logger.info(f"[{idx}/{len(voi_files)}] Converting: {voi_file.name}")
            
            # Determine output path (preserve directory structure)
            relative_path = voi_file.relative_to(input_dir)
            output_path = output_dir / relative_path.parent / (relative_path.stem + ".nii")
            
            # Convert
            nifti_path = voi2nii(
                voi_file,
                output_path,
                config=config,
                verbose=verbose
            )
            
            results['successful'].append(str(nifti_path))
            logger.info(f"✓ Success: {voi_file.name}")
            
        except Exception as e:
            logger.error(f"✗ Failed to convert {voi_file.name}: {e}")
            results['failed'].append(str(voi_file))
    
    # Summary
    logger.info("="*60)
    logger.info(f"Batch conversion complete:")
    logger.info(f"  Successful: {len(results['successful'])}")
    logger.info(f"  Failed: {len(results['failed'])}")
    logger.info(f"  Total: {len(voi_files)}")
    logger.info("="*60)
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("VOI to NIfTI Conversion Module imported successfully")