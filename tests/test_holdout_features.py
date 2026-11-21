import csv
from pathlib import Path


DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "raw" / "holdout_features.csv"


def test_holdout_features_is_readable():
    with DATA_FILE.open(newline="", encoding="utf-8") as data:
        reader = csv.DictReader(data)
        rows = list(reader)

    assert rows
    assert reader.fieldnames == [
        "record_id",
        "date",
        "students_on_campus",
        "menu_price_rs",
        "temp_c",
        "exam_days_left",
        "special_dishes",
        "queue_minutes_yesterday",
        "day_type",
        "block",
    ]
