import os
import shutil
from torchvision import datasets, transforms
from torch.utils.data import random_split
from config import RAW_THERMAL_DATASET_DIR, CLEANED_THERMAL_DATASET_DIR
from processors.base import ProcessDataBase
from collections import defaultdict
from tqdm import tqdm

class ThermalDatasetProcessor(ProcessDataBase):
    raw_data_path = RAW_THERMAL_DATASET_DIR
    output_base_path = CLEANED_THERMAL_DATASET_DIR

    def handle_data_processing(self):
        """
        Process raw thermal classification images from multiple roots into
        train, validation, and test splits.
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print("⏳ Merging classification roots...")

        merged_dir = os.path.join(self.output_base_path, '_merged_temp')
        if os.path.exists(merged_dir):
            shutil.rmtree(merged_dir)
        os.makedirs(merged_dir, exist_ok=True)

        dataset = datasets.ImageFolder(self.raw_data_path)
        for path, label in tqdm(dataset.samples, desc=f"Processing {self.raw_data_path}"):
            class_name = dataset.classes[label]
            dest_dir = os.path.join(merged_dir, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            filename = os.path.basename(path)
            dest_path = os.path.join(dest_dir, filename)

            if os.path.exists(dest_path):
                base, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(dest_path):
                    new_name = f"{base}_{counter}{ext}"
                    dest_path = os.path.join(dest_dir, new_name)
                    counter += 1
            shutil.copy(path, dest_path)

        full_dataset = datasets.ImageFolder(root=merged_dir, transform=transform)
        print(f"✅ Merged dataset has {len(full_dataset)} images across {len(full_dataset.classes)} classes.")

        class_counts = defaultdict(int)
        for _, label in full_dataset.samples:
            class_name = full_dataset.classes[label]
            class_counts[class_name] += 1
        print("\n📊 Class Distribution:")
        for cls, count in class_counts.items():
            print(f"  {cls}: {count} images")

        total_size = len(full_dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

        split_map = {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset,
        }

        for split_name, split_data in split_map.items():
            print(f"\n✍️ Writing {split_name} split...")
            for idx in tqdm(split_data.indices, desc=f"Copying {split_name}"):
                img_path, label = full_dataset.samples[idx]
                class_name = full_dataset.classes[label]
                dest_dir = os.path.join(self.output_base_path, split_name, class_name)
                os.makedirs(dest_dir, exist_ok=True)
                filename = os.path.basename(img_path)
                dest_path = os.path.join(dest_dir, filename)

                if os.path.exists(dest_path):
                    base, ext = os.path.splitext(filename)
                    counter = 1
                    while os.path.exists(dest_path):
                        new_name = f"{base}_{counter}{ext}"
                        dest_path = os.path.join(dest_dir, new_name)
                        counter += 1

                shutil.copy(img_path, dest_path)

        print(f"\n✅ Dataset split complete. Cleaned data saved to: {self.output_base_path}")
        shutil.rmtree(merged_dir)