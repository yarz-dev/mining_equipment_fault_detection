import numpy as np
from pathlib import Path
import logging

def verify_processed_data(data_dir: str) -> None:
    """
    Verify processed numpy arrays
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    data_path = Path(data_dir)
    
    for npy_file in data_path.rglob("*.npy"):
        try:
            data = np.load(str(npy_file))
            logger.info(f"✓ {npy_file.name}: shape={data.shape}, dtype={data.dtype}")
        except Exception as e:
            logger.error(f"✗ {npy_file}: {str(e)}")

if __name__ == "__main__":
    verify_processed_data("datasets/cleaned/acoustic")