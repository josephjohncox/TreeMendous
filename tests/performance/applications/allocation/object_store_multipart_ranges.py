"""Actual multipart completion smoke benchmark."""

from time import perf_counter

from treemendous.applications.allocation.multipart_upload import MultipartUploadTracker


def run_smoke(operations: int = 1000) -> dict[str, int | float]:
    if operations <= 0:
        raise ValueError("operations must be positive")
    upload = MultipartUploadTracker(operations * 8, 8)
    started = perf_counter()
    for part_number in range(1, operations + 1):
        upload.complete_part(part_number, f"etag-{part_number}")
    snapshot = upload.snapshot()
    return {
        "operations": operations,
        "completed": snapshot.diagnostics.completed_parts,
        "seconds": perf_counter() - started,
    }


if __name__ == "__main__":
    print(run_smoke())
