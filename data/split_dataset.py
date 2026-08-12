import os
import shutil
import random

# ---- Settings ----
RAW_DIR = os.path.join("data", "raw")
OUTPUT_DIR = os.path.join("data", "dataset")
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)  # so the split is reproducible

def split_dataset():
    class_names = [d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))]
    print(f"Found {len(class_names)} classes: {class_names}")

    for class_name in class_names:
        class_path = os.path.join(RAW_DIR, class_name)
        images = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)

        train_files = images[:n_train]
        val_files = images[n_train:n_train + n_val]
        test_files = images[n_train + n_val:]

        for split_name, files in [("train", train_files), ("validation", val_files), ("test", test_files)]:
            split_class_dir = os.path.join(OUTPUT_DIR, split_name, class_name)
            os.makedirs(split_class_dir, exist_ok=True)

            for fname in files:
                src = os.path.join(class_path, fname)
                dst = os.path.join(split_class_dir, fname)
                shutil.copyfile(src, dst)

        print(f"{class_name}: {n_total} total -> train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    print("\nDone! Your dataset is split into data/dataset/train, validation, and test.")

if __name__ == "__main__":
    split_dataset()