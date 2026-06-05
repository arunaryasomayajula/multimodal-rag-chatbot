"""
Migration: adds user_id column to documents and chunks tables (nullable FK).
Safe to run on existing databases — adds column only if missing.
Run once after upgrading to v0.2.0:
    python scripts/migrate_add_user_id.py
"""
from sqlalchemy import text
from db.session import engine


def _column_exists(conn, table: str, column: str) -> bool:
    result = conn.execute(text(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = :t AND column_name = :c"
    ), {"t": table, "c": column})
    return result.fetchone() is not None


def migrate():
    with engine.begin() as conn:
        for table in ("documents", "chunks"):
            if not _column_exists(conn, table, "user_id"):
                conn.execute(text(
                    f"ALTER TABLE {table} ADD COLUMN user_id VARCHAR REFERENCES users(id)"
                ))
                print(f"  Added user_id to {table}")
            else:
                print(f"  {table}.user_id already exists, skipping")
    print("Migration complete.")


if __name__ == "__main__":
    migrate()
