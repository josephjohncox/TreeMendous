"""Track out-of-order multipart completion and an ETag retry."""

from treemendous.applications.allocation.multipart_upload import MultipartUploadTracker


def main() -> None:
    upload = MultipartUploadTracker(23, 10, upload_id="video-upload")
    upload.complete_part(2, "etag-two")
    print("missing", upload.snapshot().missing_ranges)
    upload.complete_part(1, "etag-one")
    upload.complete_part(2, "etag-two-retry", retry=True)
    print("contiguous", upload.snapshot().contiguous_completion)


if __name__ == "__main__":
    main()
