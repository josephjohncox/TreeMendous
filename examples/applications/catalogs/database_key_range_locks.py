"""Lock a bytewise ordered database key band."""

from treemendous.applications.catalogs.database_key_range_locks import (
    DatabaseKeyRangeLocks,
    LockMode,
)


def main() -> None:
    locks = DatabaseKeyRangeLocks()
    handle = locks.acquire("accounts", "transaction-7", "a", "m", LockMode.EXCLUSIVE)
    print(locks.snapshot().locks)
    locks.release(handle, owner="transaction-7")


if __name__ == "__main__":
    main()
