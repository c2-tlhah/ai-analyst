import json
import sqlite3

from scripts import build_database


def test_generic_builder_discovers_arbitrary_csvs_and_relationships(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "authors.csv").write_text(
        "author_id,display_name\n1,Ada\n2,Grace\n", encoding="utf-8"
    )
    (source / "articles.csv").write_text(
        "article_id,author_id,published_at,word_count\n"
        "10,1,2025-01-01,500\n11,2,2025-01-02,700\n",
        encoding="utf-8",
    )
    output = tmp_path / "unseen.db"

    assert build_database.build(input_dir=source, output=output) == output.resolve()

    conn = sqlite3.connect(output)
    assert {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    } == {"authors", "articles"}
    assert conn.execute("PRAGMA table_info('authors')").fetchall()[0][5] == 1
    fk = conn.execute("PRAGMA foreign_key_list('articles')").fetchone()
    assert (fk[2], fk[3], fk[4]) == ("authors", "author_id", "author_id")
    conn.close()


def test_generic_builder_manifest_controls_names_types_and_required_rows(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "raw.csv").write_text(
        "Device Code,Recorded Value\nd-1,2.5\n,9.0\n", encoding="utf-8"
    )
    manifest = tmp_path / "schema.json"
    manifest.write_text(
        json.dumps(
            {
                "tables": {
                    "Sensor Readings": {
                        "source": "raw.csv",
                        "rename": {
                            "Device Code": "device-code",
                            "Recorded Value": "reading.value",
                        },
                        "types": {"reading.value": "NUMERIC"},
                        "required": ["device-code"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "manifest.db"

    build_database.build(input_dir=source, output=output, manifest_path=manifest)

    conn = sqlite3.connect(output)
    assert conn.execute('SELECT COUNT(*) FROM "Sensor Readings"').fetchone()[0] == 1
    columns = conn.execute("PRAGMA table_info('Sensor Readings')").fetchall()
    assert [(row[1], row[2]) for row in columns] == [
        ("device-code", "TEXT"),
        ("reading.value", "NUMERIC"),
    ]
    conn.close()
