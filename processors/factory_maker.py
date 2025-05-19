from .base import ProcessDataBase
from .types import ProcessorKinds

def make_factory_objects(kind: ProcessorKinds) -> ProcessDataBase:
    """
    Factory function to create data processing objects based on the specified kind.
    
    Args:
        kind (ProcessorKinds): The type of data processing object to create. Options include 'thermal', etc.
    
    Returns:
        ProcessDataBase: An instance of a data processing class.
    """
    if kind == "thermal":
        from processors.thermal import ThermalDatasetProcessor
        return ThermalDatasetProcessor()
    if kind == "acoustic":
        from processors.acoustic import AcousticDatasetProcessor
        return AcousticDatasetProcessor()
    else:
        raise ValueError(f"Unknown kind: {kind}")