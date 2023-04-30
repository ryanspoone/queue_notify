import os
import random
import shutil
from pathlib import Path


def count_images_in_directory(directory):
    return sum([len(files) for _, _, files in os.walk(directory)])


def move_images(src, dest, num_images_to_move):
    src_files = [f for f in src.iterdir() if f.is_file()]
    random.shuffle(src_files)
    num_images_to_move = min(num_images_to_move, len(src_files))

    for i in range(num_images_to_move):
        file_to_move = src_files[i]
        destination_path = dest / file_to_move.name

        if destination_path.exists():
            file_base, file_ext = os.path.splitext(file_to_move.name)
            counter = 1
            while destination_path.exists():
                new_file_name = f"{file_base}_{counter}{file_ext}"
                destination_path = dest / new_file_name
                counter += 1

        shutil.move(str(file_to_move), destination_path)


def balance_train_val(
    train_dir, val_dir, category, target_total_images, ideal_train_ratio
):
    print(f"Processing category: {category}")

    train_dir_category = train_dir / category
    val_dir_category = val_dir / category

    train_count = count_images_in_directory(train_dir_category)
    print(f"Train images for {category}: {train_count}")
    val_count = count_images_in_directory(val_dir_category)
    print(f"Val images for {category}: {val_count}")

    total_count = train_count + val_count

    if total_count == 0:
        print(f"No images found for category: {category}")
        return

    ideal_train_count = int(target_total_images * ideal_train_ratio)
    ideal_val_count = target_total_images - ideal_train_count

    print(
        f"Initial total images for {category}: Train: {train_count}, Val: {val_count}"
    )

    if train_count > ideal_train_count:
        num_images_to_move = train_count - ideal_train_count
        print(
            f"Moving {num_images_to_move} images from train to val for category {category}"
        )
        move_images(train_dir_category, val_dir_category, num_images_to_move)
    elif train_count < ideal_train_count:
        num_images_to_move = ideal_train_count - train_count
        print(
            f"Moving {num_images_to_move} images from val to train for category {category}"
        )
        move_images(val_dir_category, train_dir_category, num_images_to_move)

    val_count = count_images_in_directory(val_dir_category)
    print(f"Val images for {category} after balancing: {val_count}")

    if val_count > ideal_val_count:
        move_images(
            val_dir / category,
            train_dir / category,
            val_count - ideal_val_count,
        )
    # If there are fewer images in the val set than the ideal number, move images from the train set to the val set
    elif val_count < ideal_val_count:
        move_images(
            train_dir / category,
            val_dir / category,
            ideal_val_count - val_count,
        )

    # Display the final number of images in the train and val sets for the category after balancing
    print(
        f"Final total images for {category}: Train: {count_images_in_directory(train_dir / category)}, "
        f"Val: {count_images_in_directory(val_dir / category)}"
    )


def balance_categories(train_dir, val_dir, categories):
    # Calculate the total number of images across all categories by summing the count of images in
    # the train and val directories for each category.
    total_images = sum(
        [
            count_images_in_directory(train_dir / category)
            + count_images_in_directory(val_dir / category)
            for category in categories
        ]
    )

    # Calculate the ideal number of images per category for a balanced distribution by dividing
    # the total number of images by the number of categories.
    ideal_images_per_category = total_images // len(categories)

    # Iterate through each category to balance the total number of images between categories.
    for category in categories:
        # Get the current count of images in the train and val directories for the category.
        current_train_count = count_images_in_directory(train_dir / category)
        current_val_count = count_images_in_directory(val_dir / category)

        # Calculate the current total count of images for the category.
        current_total_count = current_train_count + current_val_count

        # If the current total count is greater than the ideal count, move the excess images
        # from the train directory to the val directory.
        if current_total_count > ideal_images_per_category:
            move_images(
                train_dir / category,
                val_dir / category,
                current_total_count - ideal_images_per_category,
            )
        # If the current total count is less than the ideal count, move the required number of
        # images from the val directory to the train directory to achieve the ideal count.
        elif current_total_count < ideal_images_per_category:
            move_images(
                val_dir / category,
                train_dir / category,
                ideal_images_per_category - current_total_count,
            )

    # Iterate through each category to balance the train and val sets for the category using
    # the updated total count for each category.
    for category in categories:
        # Get the current count of images in the train and val directories for the category.
        current_train_count = count_images_in_directory(train_dir / category)
        current_val_count = count_images_in_directory(val_dir / category)

        # Calculate the current total count of images for the category.
        current_total_count = current_train_count + current_val_count

        # Call the balance_train_val function with the current total count for the category and
        # an ideal_train_ratio of 0.8 (80% of images in the train set, 20% in the val set).
        balance_train_val(
            train_dir,
            val_dir,
            category,
            current_total_count,
            ideal_train_ratio=0.8,
        )


def move_screenshots_to_train_val(
    screenshots_dir, train_dir, val_dir, categories
):
    for category in categories:
        count_train = count_images_in_directory(train_dir / category)
        count_val = count_images_in_directory(val_dir / category)
        total = count_train + count_val
        ideal_train_count = int(total * 0.8)
        num_images_to_move = ideal_train_count - count_train
        if num_images_to_move > 0:
            print(
                f"Moving {num_images_to_move} images from screenshots to train for category {category}"
            )
            move_images(
                screenshots_dir / category,
                train_dir / category,
                num_images_to_move,
            )
        elif num_images_to_move < 0:
            print(
                f"Moving {abs(num_images_to_move)} images from train to screenshots for category {category}"
            )
            move_images(
                train_dir / category,
                screenshots_dir / category,
                abs(num_images_to_move),
            )


def main():
    project_root = Path(__file__).resolve().parent.parent
    train_dir = project_root / "train"
    val_dir = project_root / "val"
    screenshots_dir = project_root / "screenshots"
    categories = ["queue_pop", "not_queue_pop"]

    print("Moving all images from screenshots to train directories...")
    for category in categories:
        move_images(
            screenshots_dir / category,
            train_dir / category,
            count_images_in_directory(screenshots_dir / category),
        )

    print("Balancing categories...")
    balance_categories(train_dir, val_dir, categories)

    print("\nFinal image counts:")
    for category in categories:
        print(
            f"{category}: Train: {count_images_in_directory(train_dir / category)}, "
            f"Val: {count_images_in_directory(val_dir / category)}"
        )


if __name__ == "__main__":
    main()
