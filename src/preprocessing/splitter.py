import re
import shutil
import pandas as pd
from pathlib import Path
from utils.logger import setup_logger
from collections import defaultdict
from sklearn.model_selection import train_test_split


logger = setup_logger(__name__)
FILENAME = re.compile(
    r"(?P<subject>s\d+)_(?P<attire>civ|uni)(?P<variant>\d+)_(?P<view>bv|fv)_(?P<position>lp|cp|rp)_(?P<bag>nb|wb)(?:_(?P<eyeglasses>neg|weg))?_v(?P<repetition>\d+)"
)


def group_videos(dataset: Path):
    groups = defaultdict(list)

    for video in Path(dataset).rglob("*.mp4"):
        if not (match := FILENAME.match(video.stem)):
            continue

        key = match.groupdict()
        name = (
            f"{key['subject']}_{key['view']}"
            if key["attire"] == "civ" and key["variant"] == "1"
            else "_".join(
                filter(
                    None,
                    [
                        key["subject"],
                        f"{key['attire']}{key['variant']}",
                        key["view"],
                        key["position"],
                        key["bag"],
                        key.get("eyeglasses", ""),
                    ],
                )
            )
        )
        groups[name].append({"path": video, **key})

    logger.info(f"Grouped {len(groups)} video groups.")
    return groups


def split_videos(dataset: Path):
    groups = group_videos(dataset)

    data_list = [
        {**item, "key": key, "eyeglasses": item.get("eyeglasses", "")}
        for key, items in groups.items()
        for item in items
    ]
    test_size = min(
        max(0.3, len(set(d["key"] for d in data_list)) / len(data_list)), 0.4
    )

    train_data, test_data = train_test_split(
        data_list,
        test_size=test_size,
        stratify=[d["key"] for d in data_list],
        random_state=42,
    )
    val_data, final_test_data = train_test_split(
        test_data,
        test_size=0.5,
        stratify=[(d["subject"], d["attire"], d["view"], d["bag"]) for d in test_data],
        random_state=42,
    )

    logger.info(
        f"Split data into {len(train_data)} train, {len(val_data)} validation, and {len(final_test_data)} test videos."
    )
    return train_data, val_data, final_test_data


def save_split(dataset: Path, outpath: Path):
    train, val, test = split_videos(dataset)
    splits = {"train": train, "val": val, "test": test}

    summary = []
    for split, videos in splits.items():
        for key in videos:
            dest = outpath / split / key["subject"] / key["view"] / key["attire"]
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy(key["path"], dest / key["path"].name)
            summary.append(
                [
                    key[k]
                    for k in [
                        "path",
                        "subject",
                        "attire",
                        "variant",
                        "view",
                        "position",
                        "bag",
                        "eyeglasses",
                        "repetition",
                    ]
                ]
                + [split]
            )

    df = pd.DataFrame(
        summary,
        columns=[
            "path",
            "subject",
            "attire",
            "variant",
            "view",
            "position",
            "bag",
            "eyeglasses",
            "repetition",
            "split",
        ],
    )
    df.to_csv(outpath / "summary.csv", index=False)
    logger.info(f"Summary saved to {outpath} folder.")


if __name__ == "__main__":
    save_split("dataset/raw", Path("dataset/splits"))
