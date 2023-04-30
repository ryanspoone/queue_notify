import os
from pathlib import Path
from datetime import datetime
import itertools


def unique_image_names(directory):
    images = itertools.chain(
        directory.glob("*.[jJ][pP][gG]"),
        directory.glob("*.[pP][nN][gG]"),
    )

    for idx, image_path in enumerate(sorted(images)):
        creation_time = datetime.fromtimestamp(os.path.getctime(image_path))
        new_image_path = (
            image_path.parent
            / f"screenshots_{creation_time.strftime('%Y%m%d_%H%M%S')}_{idx}{image_path.suffix}"
        )
        image_path.rename(new_image_path)


def main():
    project_root = Path(__file__).resolve().parent.parent
    train_dir = project_root / "train"
    val_dir = project_root / "val"
    categories = ["queue_pop", "not_queue_pop"]

    for category in categories:
        print(f"Renaming images in train/{category}...")
        unique_image_names(train_dir / category)

        print(f"Renaming images in val/{category}...")
        unique_image_names(val_dir / category)


if __name__ == "__main__":
    main()
