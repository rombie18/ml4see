"""Various utility methods"""


def chunker(seq, size):
    """Helper method for iterating over chunks of a list"""
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def require_processing_stage(h5file, stage_req, strict=False):
    """Raises an exception when a file does not undergone the required level of processing."""
    file_stage = int(h5file["meta"].attrs["processing_stage"])

    if not strict:
        if file_stage < stage_req:
            raise ValueError(
                f"HDF file is not at required level of processing. Found: {file_stage}; Required: >={stage_req}."
            )
    else:
        if file_stage != stage_req:
            raise ValueError(
                f"HDF file is not at required level of processing. Found: {file_stage}; Required: =={stage_req}."
            )
