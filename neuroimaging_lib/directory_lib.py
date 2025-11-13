#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized file utility functions for neuroimaging pipelines.

Created on Mon Apr  7 19:12:45 2025
@author: temuuleu
"""

import logging
import re
from pathlib import Path
from typing import List, Tuple, Union, Optional

import ants  # For ANTs bias field correction
import nibabel as nib  # For handling NIfTI images
from nipype.interfaces.fsl import BET  # For brain extraction using FSL's BET
from nipype import Node  # To run BET as a node


def get_files_from_dir(
    path: Union[str, Path],
    endings: List[str] = None,
    session_basename: Optional[str] = None,
    max_depth: Optional[int] = None,
) -> List[str]:
    """
    Recursively fetch files from a directory with specific endings.
    
    Uses pathlib for better performance and cleaner code.

    Args:
        path: Directory path to search.
        endings: File extensions to match (default: [".nii", ".nii.gz"]).
        session_basename: Substring the filename should contain (None = no filter).
        max_depth: Maximum directory depth to search (None = unlimited).

    Returns:
        List of matching file paths as strings.
    """
    if endings is None:
        endings = [".nii", ".nii.gz"]
    
    path = Path(path)
    
    if not path.is_dir():
        logging.warning(f"Provided path: {path} is not a directory.")
        return []

    base_depth = len(path.parts)
    matching_files = []

    for file_path in path.rglob("*"):
        # Check depth constraint
        if max_depth is not None and len(file_path.parts) - base_depth >= max_depth:
            continue
        
        # Check if file has matching extension
        if file_path.is_file() and any(file_path.name.endswith(ending) for ending in endings):
            # Check basename constraint
            if session_basename is None or session_basename in file_path.name:
                matching_files.append(str(file_path))

    return matching_files


def get_subdirectories(
    path: Union[str, Path],
    index: bool = False,
    basename: bool = False,
    only_num: bool = True,
    verbose: bool = False,
) -> Union[List[str], List[Tuple[int, str]]]:
    """
    Retrieve subdirectories within a given directory.
    
    Optimized with pathlib and removed duplicate code.

    Args:
        path: Path to the directory.
        index: If True, returns tuples (index, path).
        basename: If True, returns only basenames.
        only_num: If True, includes only subdirectories with digits in name.
        verbose: If True, prints directory information.

    Returns:
        List of subdirectory paths or tuples if index=True.
    """
    path = Path(path)
    
    if not path.is_dir():
        logging.warning(f"Provided path: {path} does not exist or is not a directory.")
        return []

    # Get subdirectories
    subdirs = [d for d in path.iterdir() if d.is_dir()]
    
    # Filter by digit constraint if requested
    if only_num:
        subdirs = [d for d in subdirs if re.search(r"\d", d.name)]
    
    # Convert to appropriate format
    data_paths_list = [d.name if basename else str(d) for d in subdirs]
    data_paths_list.sort()
    
    if verbose:
        print(f"path: {path} exists")
        print(f"number of paths: {len(data_paths_list)}")
    
    # Return with or without index
    if index:
        return [(i, p) for i, p in enumerate(data_paths_list)]
    
    return data_paths_list


def find_elements(
    file_list: List[str],
    include: List[str] = None,
    exclude: List[str] = None,
    case_sensitive: bool = False,
) -> List[str]:
    """
    Filter files based on inclusion and exclusion criteria.
    
    Optimized: convert case once before filtering.

    Args:
        file_list: List of file paths to filter.
        include: Substrings to include (empty = include all).
        exclude: Substrings to exclude.
        case_sensitive: Whether to preserve case sensitivity.

    Returns:
        Filtered list of files.
    """
    if include is None:
        include = []
    if exclude is None:
        exclude = []
    
    # Normalize case once
    if not case_sensitive:
        include = [inc.lower() for inc in include]
        exclude = [exc.lower() for exc in exclude]

    filtered_files = []
    
    for file in file_list:
        basename = Path(file).name
        if not case_sensitive:
            basename = basename.lower()
        
        # Check inclusion condition
        include_condition = any(inc in basename for inc in include) if include else True
        
        # Check exclusion condition
        exclude_condition = not any(exc in basename for exc in exclude)
        
        if include_condition and exclude_condition:
            filtered_files.append(file)

    return filtered_files


def find_one_element(
    file_list: List[str],
    include: List[str] = None,
    exclude: List[str] = None,
    case_sensitive: bool = False,
) -> str:
    """
    Find exactly one file matching inclusion/exclusion criteria.
    
    Optimized: reuses find_elements logic to avoid duplication.

    Args:
        file_list: List of file paths to filter.
        include: Substrings to include (empty = include all).
        exclude: Substrings to exclude.
        case_sensitive: Whether to preserve case sensitivity.

    Returns:
        The single matching file path.

    Raises:
        ValueError: If zero or more than one file matches criteria.
    """
    if include is None:
        include = []
    if exclude is None:
        exclude = []
    
    filtered_files = find_elements(file_list, include, exclude, case_sensitive)
    
    if len(filtered_files) == 0:
        logging.error("No files found matching the specified criteria")
        raise ValueError("No files found matching the specified criteria")
    elif len(filtered_files) > 1:
        logging.error(f"Multiple files found ({len(filtered_files)}): {filtered_files}")
        raise ValueError(f"Multiple files found ({len(filtered_files)}): {filtered_files}")
    
    logging.info(f"Found file: {filtered_files[0]}")
    return filtered_files[0]