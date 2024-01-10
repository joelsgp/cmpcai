import csv
import re
from datetime import datetime
from io import StringIO
from pathlib import Path

from common import DATA_TEXT_PATH

file_re = re.compile(
    "home of eggyboi - (?P<category>.+) - (?P<channel>.+) \[\d+\]\.csv"
)

# smsg - start message
# emsg - end message
template = """<|smsg|>
Category: {category}
Channel: {channel}
Date: {date}
Time: {time}
User: {user}
Message:
{message}
<|emsg|>
"""


def format_date(timestamp: str) -> tuple[str, str]:
    dt = datetime.fromisoformat(timestamp)
    date = dt.strftime("%A, %d %B %Y-%m-%d")
    time = dt.strftime("%H:%M")
    return date, time


def format_file(category: str, channel: str, reader: csv.DictReader, outfile: StringIO):
    for row in reader:
        user = row["Author"]
        if user == "Deleted User":
            user += row["AuthorID"]

        timestamp = row["Date"]
        date, time = format_date(timestamp)

        message = row["Content"]

        train_message = template.format(
            category=category,
            channel=channel,
            user=user,
            date=date,
            time=time,
            message=message,
        )

        outfile.write(train_message)
    outfile.write("<|endoftext|>")


def main():
    out_dir = Path("out/")
    train_file = Path(DATA_TEXT_PATH)

    with open(train_file, "w", encoding="utf-8") as outfile:
        for fp in out_dir.iterdir():
            if not fp.is_file():
                continue

            print(f"doing {fp}")

            match = file_re.fullmatch(fp.name)
            category = match.group("category")
            channel = match.group("channel")

            with open(fp, encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f, dialect=csv.unix_dialect)
                format_file(
                    category=category, channel=channel, reader=reader, outfile=outfile
                )


if __name__ == "__main__":
    main()
