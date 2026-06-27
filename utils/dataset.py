from datasets import load_dataset


def load_query_data(set_type, name="osunlp/TravelPlanner"):
    """Load the TravelPlanner query split (train/validation/test).

    Config name and split name both equal set_type, so
    load_dataset(name, set_type)[set_type] is the canonical pattern used
    throughout the TravelPlanner codebase.
    """
    if set_type not in ("train", "validation", "test"):
        raise ValueError(f"Unknown set_type: {set_type!r}. Must be one of train/validation/test.")
    return load_dataset(name, set_type)[set_type]
