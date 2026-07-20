"""Acquire compatible readers and release by explicit owner."""

from treemendous.applications.catalogs.filesystem_byte_locks import (
    FilesystemByteLocks,
    LockMode,
)


def main() -> None:
    locks = FilesystemByteLocks()
    handle = locks.acquire("archive.dat", "reader", 0, 4096, LockMode.SHARED)
    print(locks.active("archive.dat", 100, 200))
    locks.release(handle, owner="reader")


if __name__ == "__main__":
    main()
