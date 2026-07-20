"""Multipart upload application contracts."""

from dataclasses import replace
from typing import cast

import pytest

from tests.oracles.applications.allocation.object_store_multipart_ranges import state
from treemendous.applications.allocation.multipart_upload import (
    MultipartUploadTracker,
    PartConflictError,
)
from treemendous.domain import Span


def test_out_of_order_parts_missing_ranges_and_contiguous_prefix() -> None:
    upload = MultipartUploadTracker(23, 10, upload_id="u")
    upload.complete_part(2, "etag-2")
    expected_missing, expected_contiguous = state(23, 10, {2})
    assert upload.snapshot().missing_ranges == expected_missing
    assert upload.snapshot().contiguous_completion == expected_contiguous
    upload.complete_part(1, "etag-1")
    assert upload.snapshot().contiguous_completion == Span(0, 20)
    assert upload.complete_part(3, "etag-3", size=3).byte_range == Span(20, 23)


def test_etag_retries_identity_checkpoint_and_atomic_failures() -> None:
    upload = MultipartUploadTracker(20, 10)
    original = upload.complete_part(1, "old")
    assert upload.complete_part(1, "old") is original
    before = upload.snapshot()
    with pytest.raises(PartConflictError):
        upload.complete_part(1, "new")
    assert upload.snapshot() == before
    retried = upload.complete_part(1, "new", retry=True)
    assert retried.attempt == 2
    checkpoint = upload.checkpoint()
    upload.complete_part(2, "second")
    upload.restore(checkpoint)
    assert upload.snapshot().diagnostics.retry_count == 1


def test_restore_rejects_forged_reserved_geometry_atomically() -> None:
    upload = MultipartUploadTracker(20, 10)
    checkpoint = upload.checkpoint()
    forged_allocator = replace(
        checkpoint.allocator,
        reserved_ranges=(Span(0, 5),),
        free_ranges=(Span(5, 20),),
    )
    before = upload.snapshot()

    with pytest.raises(ValueError, match="configured allocator geometry"):
        upload.restore(replace(checkpoint, allocator=forged_allocator))

    assert upload.snapshot() == before
    assert upload.complete_part(1, "etag").byte_range == Span(0, 10)


def test_restore_rejects_non_string_etag_atomically() -> None:
    upload = MultipartUploadTracker(20, 10)
    upload.complete_part(1, "etag")
    checkpoint = upload.checkpoint()
    forged_part = replace(checkpoint.completed[0], etag=cast(str, 7))
    before = upload.snapshot()

    with pytest.raises(ValueError, match="completed-part metadata"):
        upload.restore(replace(checkpoint, completed=(forged_part,)))

    assert upload.snapshot() == before
