"""
platform_utils.py - Platform detection utilities for MPS tests

Provides utilities for detecting Apple Silicon chip generation, features,
and creating platform-aware test decorators.
"""

import functools
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class PlatformInfo:
    """Platform information for a test run."""
    chip_name: str
    chip_generation: int  # 1=M1, 2=M2, 3=M3, 4=M4, 0=unknown
    gpu_cores: int
    memory_gb: float
    macos_version: str
    has_dynamic_caching: bool
    is_ultra: bool

    def to_dict(self) -> dict:
        return {
            "chip_name": self.chip_name,
            "chip_generation": self.chip_generation,
            "gpu_cores": self.gpu_cores,
            "memory_gb": self.memory_gb,
            "macos_version": self.macos_version,
            "has_dynamic_caching": self.has_dynamic_caching,
            "is_ultra": self.is_ultra
        }


def get_chip_name() -> str:
    """Get the Apple Silicon chip name."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "Unknown"


def get_chip_generation() -> int:
    """
    Detect Apple Silicon generation.
    Returns: 1=M1, 2=M2, 3=M3, 4=M4, 0=unknown
    """
    chip = get_chip_name()
    if "M1" in chip:
        return 1
    if "M2" in chip:
        return 2
    if "M3" in chip:
        return 3
    if "M4" in chip:
        return 4
    return 0


def is_ultra_chip() -> bool:
    """Check if running on an Ultra variant (dual-die)."""
    return "Ultra" in get_chip_name()


def has_dynamic_caching() -> bool:
    """Check if chip has Dynamic Caching (M3+)."""
    return get_chip_generation() >= 3


def get_gpu_core_count() -> int:
    """Get the number of GPU cores."""
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, check=True, timeout=15
        )
        for line in result.stdout.split("\n"):
            if "Total Number of Cores:" in line:
                parts = line.split(":")
                if len(parts) >= 2:
                    return int(parts[1].strip())
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        pass
    return 0


def get_memory_gb() -> float:
    """Get total system memory in GB."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, check=True
        )
        bytes_val = int(result.stdout.strip())
        return bytes_val / (1024 ** 3)
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        return 0.0


def get_macos_version() -> str:
    """Get macOS version string."""
    return platform.mac_ver()[0]


def get_platform_info() -> PlatformInfo:
    """Get comprehensive platform information."""
    return PlatformInfo(
        chip_name=get_chip_name(),
        chip_generation=get_chip_generation(),
        gpu_cores=get_gpu_core_count(),
        memory_gb=get_memory_gb(),
        macos_version=get_macos_version(),
        has_dynamic_caching=has_dynamic_caching(),
        is_ultra=is_ultra_chip()
    )


def print_platform_info():
    """Print platform information to stdout."""
    info = get_platform_info()
    print(f"Platform: {info.chip_name}")
    print(f"  Generation: M{info.chip_generation}" if info.chip_generation else "  Generation: Unknown")
    print(f"  GPU Cores: {info.gpu_cores}")
    print(f"  Memory: {info.memory_gb:.1f} GB")
    print(f"  macOS: {info.macos_version}")
    print(f"  Dynamic Caching: {'Yes' if info.has_dynamic_caching else 'No'}")
    print(f"  Ultra (dual-die): {'Yes' if info.is_ultra else 'No'}")


# Test decorators for platform-specific tests

def requires_generation(min_gen: int):
    """Skip test if chip generation is below minimum."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            gen = get_chip_generation()
            if gen < min_gen:
                import pytest
                pytest.skip(f"Requires M{min_gen}+ (current: M{gen if gen else '?'})")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def requires_dynamic_caching(func):
    """Skip test if Dynamic Caching not available (requires M3+)."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not has_dynamic_caching():
            import pytest
            pytest.skip("Requires M3+ with Dynamic Caching")
        return func(*args, **kwargs)
    return wrapper


def requires_ultra(func):
    """Skip test if not running on Ultra chip."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not is_ultra_chip():
            import pytest
            pytest.skip("Requires Ultra chip (dual-die)")
        return func(*args, **kwargs)
    return wrapper


def requires_gpu_cores(min_cores: int):
    """Skip test if GPU core count is below minimum."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cores = get_gpu_core_count()
            if cores < min_cores:
                import pytest
                pytest.skip(f"Requires {min_cores}+ GPU cores (current: {cores})")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def requires_memory_gb(min_gb: float):
    """Skip test if memory is below minimum."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            mem = get_memory_gb()
            if mem < min_gb:
                import pytest
                pytest.skip(f"Requires {min_gb}+ GB memory (current: {mem:.1f} GB)")
            return func(*args, **kwargs)
        return wrapper
    return decorator


class PlatformReport:
    """
    Context manager that records platform info at test start.
    Useful for including platform context in test reports.
    """

    def __init__(self, output_file: Optional[str] = None):
        self.output_file = output_file
        self.info: Optional[PlatformInfo] = None

    def __enter__(self):
        self.info = get_platform_info()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.output_file and self.info:
            with open(self.output_file, "w") as f:
                json.dump(self.info.to_dict(), f, indent=2)
        return False


if __name__ == "__main__":
    # When run directly, print platform info
    print_platform_info()

    # Also output JSON if requested
    if "--json" in sys.argv:
        info = get_platform_info()
        print(json.dumps(info.to_dict(), indent=2))
