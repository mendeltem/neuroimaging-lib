#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized neuroimaging visualization utilities for neuroimaging pipelines.

Provides functions for:
- Automatic detection of high-quality slices
- Multi-overlay panel visualization
- BIDS-compliant output naming
- Comprehensive image analysis and plotting

Features:
- Professional logging (replaces print statements)
- Type hints and error handling
- Modular, reusable components
- Configurable colors and styles
- High-quality output (DPI, format options)
"""

import logging
from pathlib import Path
from typing import Union, Optional, List, Tuple, Dict
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND CONFIGURATION
# ============================================================================

class ColorScheme(Enum):
    """Predefined color schemes for overlays."""
    RAINBOW = "rainbow"          # Red, Green, Blue, Yellow, Magenta, Cyan, Orange
    HEAT = "heat"                # Heat map colors
    DIVERGING = "diverging"      # Blue-White-Red
    GRAYSCALE = "grayscale"      # Grayscale


@dataclass
class ColorConfig:
    """Color configuration for overlays."""
    scheme: ColorScheme = ColorScheme.RAINBOW
    
    # Rainbow colors
    rainbow_colors: List[Tuple[float, float, float]] = field(default_factory=lambda: [
        (1.0, 0.0, 0.0),      # Red
        (0.0, 1.0, 0.0),      # Green
        (0.0, 0.0, 1.0),      # Blue
        (1.0, 1.0, 0.0),      # Yellow
        (1.0, 0.0, 1.0),      # Magenta
        (0.0, 1.0, 1.0),      # Cyan
        (1.0, 0.5, 0.0),      # Orange
        (0.5, 0.0, 1.0),      # Purple
        (1.0, 0.0, 0.5),      # Pink
    ])
    
    # Heat map colors (blue -> cyan -> yellow -> red)
    heat_colors: List[Tuple[float, float, float]] = field(default_factory=lambda: [
        (0.0, 0.0, 1.0),      # Blue
        (0.0, 1.0, 1.0),      # Cyan
        (1.0, 1.0, 0.0),      # Yellow
        (1.0, 0.0, 0.0),      # Red
    ])
    
    # Diverging colors (blue -> white -> red)
    diverging_colors: List[Tuple[float, float, float]] = field(default_factory=lambda: [
        (0.0, 0.0, 1.0),      # Blue
        (1.0, 1.0, 1.0),      # White
        (1.0, 0.0, 0.0),      # Red
    ])
    
    overlay_alpha: float = 0.5  # Transparency of overlay (0-1)
    
    def get_colors(self, n_overlays: int) -> List[Tuple[float, float, float]]:
        """Get colors for n overlays based on scheme."""
        if self.scheme == ColorScheme.RAINBOW:
            colors = [self.rainbow_colors[i % len(self.rainbow_colors)] for i in range(n_overlays)]
        elif self.scheme == ColorScheme.HEAT:
            # Interpolate heat colors
            indices = np.linspace(0, len(self.heat_colors) - 1, n_overlays)
            colors = [self._interpolate_color(self.heat_colors, idx) for idx in indices]
        elif self.scheme == ColorScheme.DIVERGING:
            indices = np.linspace(0, len(self.diverging_colors) - 1, n_overlays)
            colors = [self._interpolate_color(self.diverging_colors, idx) for idx in indices]
        else:
            colors = [(0.5, 0.5, 0.5)] * n_overlays  # Grayscale
        
        return colors
    
    @staticmethod
    def _interpolate_color(colors: List[Tuple[float, float, float]], 
                          index: float) -> Tuple[float, float, float]:
        """Interpolate color at fractional index."""
        if index <= 0:
            return colors[0]
        if index >= len(colors) - 1:
            return colors[-1]
        
        lower_idx = int(index)
        upper_idx = lower_idx + 1
        fraction = index - lower_idx
        
        lower_color = colors[lower_idx]
        upper_color = colors[upper_idx]
        
        return tuple(
            lower_color[i] * (1 - fraction) + upper_color[i] * fraction
            for i in range(3)
        )


@dataclass
class PlotConfig:
    """Configuration for plotting."""
    dpi: int = 200
    figsize_per_panel: Tuple[float, float] = (5.0, 5.0)
    cmap_base: str = "gray"
    save_format: str = "png"
    bbox_inches: str = "tight"
    color_config: ColorConfig = field(default_factory=ColorConfig)


@dataclass
class SliceSelectionResult:
    """Result of slice selection."""
    slice_indices: List[int]
    non_zero_counts: List[int]
    valid: bool = True
    error_message: Optional[str] = None


@dataclass
class PlotResult:
    """Result of plotting operation."""
    success: bool
    output_path: Optional[Path] = None
    file_size: Optional[int] = None
    error_message: Optional[str] = None
    n_slices: int = 0
    n_panels: int = 0


# ============================================================================
# INPUT VALIDATION
# ============================================================================

def _validate_nifti_path(path: Union[str, Path], verbose: bool = False) -> Path:
    """
    Validate NIfTI file path.
    
    Parameters:
        path: Path to NIfTI file.
        verbose: If True, log validation details.
    
    Returns:
        Path: Validated path.
    
    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file is not a NIfTI file.
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {path}")
    
    if path.suffix not in ['.nii', '.gz'] and path.suffixes not in [['.nii', '.gz']]:
        logger.warning(f"File extension unexpected: {path.suffix}")
    
    if verbose:
        logger.info(f"✓ NIfTI file validated: {path.name}")
    
    return path


def _load_nifti(path: Union[str, Path]) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """
    Load NIfTI file with error handling.
    
    Parameters:
        path: Path to NIfTI file.
    
    Returns:
        Tuple[np.ndarray, nib.Nifti1Image]: Data array and nibabel image object.
    
    Raises:
        FileNotFoundError: If file not found.
        RuntimeError: If file cannot be loaded.
    """
    path = _validate_nifti_path(path)
    
    try:
        logger.debug(f"Loading NIfTI: {path.name}")
        img = nib.load(path)
        data = img.get_fdata()
        
        logger.debug(f"Loaded NIfTI shape: {data.shape}")
        return data, img
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"NIfTI file not found: {path}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to load NIfTI {path}: {e}") from e


# ============================================================================
# SLICE SELECTION
# ============================================================================

def get_top_slices(
    path: Union[str, Path],
    top_n: int = 10,
    axis: int = 2,
    verbose: bool = False
) -> SliceSelectionResult:
    """
    Identify top N slices with highest non-zero voxel counts.
    
    Optimizations:
    - Type hints and error handling
    - Configurable axis
    - Returns detailed result object
    - Professional logging

    Parameters:
        path: Path to NIfTI image.
        top_n: Number of top slices to return (default: 10).
        axis: Axis along which to select slices (default: 2 for z-axis).
        verbose: If True, show detailed information.

    Returns:
        SliceSelectionResult: Detailed result with slice indices and counts.

    Raises:
        FileNotFoundError: If file not found.
        ValueError: If image dimensions invalid.
        RuntimeError: If loading fails.

    Example:
        >>> result = get_top_slices("image.nii.gz", top_n=10)
        >>> if result.valid:
        ...     print(f"Top slices: {result.slice_indices}")
        ... else:
        ...     print(f"Error: {result.error_message}")
    """
    try:
        logger.info(f"Selecting top {top_n} slices from: {path}")
        
        # Load NIfTI
        data, img = _load_nifti(path)
        
        # Validate dimensions
        if data.ndim < 3:
            error_msg = f"Image must have 3+ dimensions, got {data.ndim}"
            logger.error(error_msg)
            return SliceSelectionResult(
                slice_indices=[],
                non_zero_counts=[],
                valid=False,
                error_message=error_msg
            )
        
        if axis >= data.ndim or axis < -data.ndim:
            error_msg = f"Invalid axis {axis} for {data.ndim}D image"
            logger.error(error_msg)
            return SliceSelectionResult(
                slice_indices=[],
                non_zero_counts=[],
                valid=False,
                error_message=error_msg
            )
        
        # Normalize axis
        if axis < 0:
            axis = data.ndim + axis
        
        n_slices = data.shape[axis]
        
        # Calculate non-zero voxel counts
        logger.debug(f"Analyzing {n_slices} slices along axis {axis}...")
        slice_info = []
        
        for z in range(n_slices):
            # Select slice along specified axis
            if axis == 0:
                slice_data = data[z, :, :]
            elif axis == 1:
                slice_data = data[:, z, :]
            else:  # axis == 2
                slice_data = data[:, :, z]
            
            non_zero_count = np.count_nonzero(slice_data)
            slice_info.append((z, non_zero_count))
        
        # Sort by non-zero count (descending)
        sorted_slices = sorted(slice_info, key=lambda x: x[1], reverse=True)
        
        # Get top N
        top_n = min(top_n, len(sorted_slices))
        top_slices_info = sorted_slices[:top_n]
        top_indices = [idx for idx, count in top_slices_info]
        top_counts = [count for idx, count in top_slices_info]
        
        if verbose:
            logger.info(f"✓ Found {len(top_indices)} top slices:")
            for i, (idx, count) in enumerate(top_slices_info[:5]):  # Show first 5
                logger.info(f"  Slice {idx}: {count} non-zero voxels")
            if len(top_slices_info) > 5:
                logger.info(f"  ... and {len(top_slices_info) - 5} more")
        
        return SliceSelectionResult(
            slice_indices=top_indices,
            non_zero_counts=top_counts,
            valid=True
        )
        
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Validation error: {e}")
        return SliceSelectionResult(
            slice_indices=[],
            non_zero_counts=[],
            valid=False,
            error_message=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error in get_top_slices: {e}", exc_info=True)
        return SliceSelectionResult(
            slice_indices=[],
            non_zero_counts=[],
            valid=False,
            error_message=str(e)
        )


# ============================================================================
# SLICE LOADING AND PROCESSING
# ============================================================================

def load_slice(
    path: Union[str, Path],
    slice_idx: int,
    axis: int = 2,
    rotate: bool = True
) -> Optional[np.ndarray]:
    """
    Load a single slice from NIfTI image.
    
    Parameters:
        path: Path to NIfTI file.
        slice_idx: Slice index.
        axis: Axis to slice along (default: 2).
        rotate: If True, rotate slice 90 degrees.
    
    Returns:
        Optional[np.ndarray]: Slice data, or None if failed.
    """
    try:
        data, _ = _load_nifti(path)
        
        # Validate slice index
        if slice_idx >= data.shape[axis] or slice_idx < 0:
            logger.warning(
                f"Slice index {slice_idx} out of range (0-{data.shape[axis]-1})"
            )
            return None
        
        # Extract slice
        if axis == 0:
            slice_data = data[slice_idx, :, :]
        elif axis == 1:
            slice_data = data[:, slice_idx, :]
        else:  # axis == 2
            slice_data = data[:, :, slice_idx]
        
        # Rotate if requested
        if rotate:
            slice_data = np.rot90(slice_data)
        
        logger.debug(f"Loaded slice {slice_idx}: shape {slice_data.shape}")
        return slice_data
        
    except Exception as e:
        logger.error(f"Failed to load slice {slice_idx} from {path}: {e}")
        return None


def normalize_slice(slice_data: np.ndarray) -> np.ndarray:
    """
    Normalize slice to 0-1 range.
    
    Parameters:
        slice_data: Input slice array.
    
    Returns:
        np.ndarray: Normalized slice (0-1 range).
    """
    data_max = np.max(slice_data)
    if data_max > 0:
        return slice_data / data_max
    return slice_data


# ============================================================================
# OVERLAY CREATION
# ============================================================================

def create_multi_overlay(
    base_slice: np.ndarray,
    overlay_slices: List[np.ndarray],
    colors: Optional[List[Tuple[float, float, float]]] = None,
    alpha: float = 0.5
) -> Optional[np.ndarray]:
    """
    Create RGB image with multiple colored overlays.
    
    Optimizations:
    - Type hints and error handling
    - Configurable colors
    - Professional logging

    Parameters:
        base_slice: Base grayscale image.
        overlay_slices: List of overlay masks.
        colors: RGB colors for each overlay.
        alpha: Overlay transparency (0-1).

    Returns:
        Optional[np.ndarray]: RGB image (H x W x 3), or None if failed.
    """
    try:
        if base_slice is None or base_slice.size == 0:
            logger.error("Base slice is None or empty")
            return None
        
        # Normalize base
        base = normalize_slice(base_slice.copy())
        
        # Create RGB from grayscale
        rgb = np.stack([base] * 3, axis=-1)
        
        # Use default colors if not provided
        if colors is None:
            config = ColorConfig()
            colors = config.get_colors(len(overlay_slices))
        
        logger.debug(f"Creating overlay with {len(overlay_slices)} masks")
        
        # Apply overlays
        for i, overlay in enumerate(overlay_slices):
            if overlay is None or overlay.size == 0:
                logger.warning(f"Overlay {i} is None or empty, skipping")
                continue
            
            # Normalize overlay
            overlay_norm = normalize_slice(overlay.copy())
            
            # Get color
            color = colors[i] if i < len(colors) else (1.0, 1.0, 1.0)
            
            # Blend overlay with base
            overlay_mask = overlay_norm > 0
            for c in range(3):
                rgb[overlay_mask, c] = (
                    alpha * color[c] * overlay_norm[overlay_mask] +
                    (1 - alpha) * rgb[overlay_mask, c]
                )
        
        # Clip to valid range
        rgb = np.clip(rgb, 0, 1)
        
        logger.debug(f"Overlay created: shape {rgb.shape}")
        return rgb
        
    except Exception as e:
        logger.error(f"Failed to create overlay: {e}", exc_info=True)
        return None


# ============================================================================
# PANEL PLOTTING
# ============================================================================

def create_panel_plot(
    images: List[Tuple[Union[str, Path], List[Union[str, Path]]]],
    output_path: Union[str, Path],
    slice_indices: List[int],
    titles: Optional[List[str]] = None,
    main_title: Optional[str] = None,
    subject_name: str = "Subject",
    config: Optional[PlotConfig] = None,
    verbose: bool = False
) -> PlotResult:
    """
    Create professional multi-panel visualization with overlays.
    
    Completely optimized with:
    - Type hints and error handling
    - Modular components
    - Professional logging (no print)
    - Configuration objects
    - Result dataclass
    - Input validation

    Parameters:
        images: List of (background_path, [overlay_paths]) tuples.
        output_path: Path to save figure.
        slice_indices: List of slice indices to display.
        titles: Titles for each panel (default: auto-generated).
        main_title: Main figure title (default: auto-generated).
        subject_name: Subject identifier.
        config: PlotConfig with visualization settings.
        verbose: If True, show detailed logging.

    Returns:
        PlotResult: Detailed result of plotting operation.

    Example:
        >>> images = [
        ...     ("brain.nii.gz", ["mask1.nii.gz", "mask2.nii.gz"]),
        ...     ("brain.nii.gz", ["mask3.nii.gz"])
        ... ]
        >>> result = create_panel_plot(
        ...     images,
        ...     "output.png",
        ...     [10, 20, 30],
        ...     titles=["Overlays", "Single Overlay"]
        ... )
        >>> if result.success:
        ...     print(f"Plot saved: {result.output_path}")
    """
    # Use default config if not provided
    if config is None:
        config = PlotConfig()
    
    logger.info("="*70)
    logger.info("Creating panel plot visualization")
    logger.info("="*70)
    
    try:
        # Validate inputs
        if not images:
            raise ValueError("No images provided")
        if not slice_indices:
            raise ValueError("No slice indices provided")
        
        n_panels = len(images)
        n_slices = len(slice_indices)
        
        logger.info(f"Configuration:")
        logger.info(f"  Panels: {n_panels}")
        logger.info(f"  Slices: {n_slices}")
        logger.info(f"  Figure size: {n_panels * config.figsize_per_panel[0]} x {n_slices * config.figsize_per_panel[1]}")
        logger.info(f"  DPI: {config.dpi}")
        logger.info(f"  Subject: {subject_name}")
        
        # Validate files
        logger.info("Validating input files...")
        for i, (bg_path, overlay_paths) in enumerate(images):
            bg_path = Path(bg_path)
            if not bg_path.exists():
                raise FileNotFoundError(f"Panel {i} background not found: {bg_path}")
            logger.debug(f"  Panel {i}: {bg_path.name} ✓")
            
            for j, overlay_path in enumerate(overlay_paths):
                overlay_path = Path(overlay_path)
                if not overlay_path.exists():
                    logger.warning(f"  Panel {i} overlay {j} not found: {overlay_path}")
        
        # Generate titles if needed
        if titles is None:
            titles = [f"Panel {i+1}" for i in range(n_panels)]
            logger.debug(f"Using auto-generated titles: {titles}")
        elif len(titles) != n_panels:
            logger.warning(f"Title count {len(titles)} != panel count {n_panels}")
            titles = titles[:n_panels] + [f"Panel {i+1}" for i in range(len(titles), n_panels)]
        
        # Generate main title if needed
        if main_title is None:
            main_title = f"{subject_name} - Top {n_slices} Slices with Overlays"
        
        logger.info(f"Title: {main_title}")
        
        # Create figure
        figsize = (
            n_panels * config.figsize_per_panel[0],
            n_slices * config.figsize_per_panel[1]
        )
        
        logger.debug(f"Creating figure: {figsize}")
        fig, axes = plt.subplots(
            n_slices, n_panels,
            figsize=figsize,
            dpi=config.dpi
        )
        
        # Handle axes shape for single row/column
        if n_slices == 1:
            axes = [axes] if n_panels == 1 else [axes]
        if n_panels == 1:
            axes = [[ax] for ax in axes]
        elif n_slices == 1:
            axes = [axes]
        
        logger.info(f"✓ Figure created")
        
        # Populate subplots
        logger.info("Populating subplots...")
        
        for i, slice_idx in enumerate(slice_indices):
            logger.debug(f"Processing slice {slice_idx} (row {i+1}/{n_slices})")
            
            for j, (bg_path, overlay_paths) in enumerate(images):
                ax = axes[i][j] if isinstance(axes[i], list) else axes[j]
                bg_path = Path(bg_path)
                
                try:
                    # Load base image
                    base = load_slice(bg_path, slice_idx)
                    if base is None:
                        logger.warning(f"Failed to load base: {bg_path.name} slice {slice_idx}")
                        ax.text(0.5, 0.5, "Error\nLoading\nBase", 
                               ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f"{titles[j]}\n[Error]", fontsize=10)
                        ax.axis("off")
                        continue
                    
                    # Load overlays
                    overlays = []
                    for overlay_path in overlay_paths:
                        overlay_path = Path(overlay_path)
                        if overlay_path.exists():
                            overlay = load_slice(overlay_path, slice_idx)
                            if overlay is not None:
                                overlays.append(overlay)
                    
                    # Create and display image
                    if overlays:
                        colors = config.color_config.get_colors(len(overlays))
                        img_data = create_multi_overlay(base, overlays, colors, 
                                                       config.color_config.overlay_alpha)
                        if img_data is not None:
                            ax.imshow(img_data)
                        else:
                            ax.imshow(base, cmap=config.cmap_base)
                    else:
                        ax.imshow(base, cmap=config.cmap_base)
                    
                    ax.set_title(f"{titles[j]}\nSlice {slice_idx}", fontsize=10)
                    ax.axis("off")
                    
                except Exception as e:
                    logger.error(f"Error in panel {j}, slice {slice_idx}: {e}")
                    ax.text(0.5, 0.5, "Display\nError", 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"{titles[j]}\n[Error]", fontsize=10)
                    ax.axis("off")
        
        # Finalize layout
        logger.debug("Finalizing plot layout...")
        plt.suptitle(main_title, fontsize=16, weight='bold')
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        
        # Save figure
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving figure to: {output_path}")
        plt.savefig(
            output_path,
            bbox_inches=config.bbox_inches,
            dpi=config.dpi,
            format=config.save_format
        )
        
        # Verify output
        if not output_path.exists():
            raise RuntimeError(f"Output file was not created: {output_path}")
        
        file_size = output_path.stat().st_size
        if file_size == 0:
            raise RuntimeError(f"Output file is empty: {output_path}")
        
        logger.info(f"✓ Plot saved successfully")
        logger.info(f"  Path: {output_path}")
        logger.info(f"  Size: {file_size / 1024:.1f} KB")
        
        return PlotResult(
            success=True,
            output_path=output_path,
            file_size=file_size,
            n_slices=n_slices,
            n_panels=n_panels
        )
        
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Validation error: {e}")
        return PlotResult(
            success=False,
            error_message=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error in create_panel_plot: {e}", exc_info=True)
        return PlotResult(
            success=False,
            error_message=str(e)
        )
    finally:
        plt.close('all')
        logger.debug("Plot closed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Plotting Module imported successfully")