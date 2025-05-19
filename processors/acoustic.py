import os
import shutil
import numpy as np
import scipy.io
from tqdm import tqdm

from config import RAW_ACOUSTIC_DATASET_DIR, CLEANED_ACOUSTIC_DATASET_DIR
from processors.base import ProcessDataBase

class AcousticDatasetProcessor(ProcessDataBase):
    raw_data_path = RAW_ACOUSTIC_DATASET_DIR
    output_base_path = CLEANED_ACOUSTIC_DATASET_DIR

    def handle_data_processing(self):
        """
        Process raw acoustic .mat files and convert them into labeled .npy files,
        then split into train, val, and test folders.
        """
        print("📥 Reading .mat files from:", self.raw_data_path)

        # Helper: recursively extract numeric array
        def extract_signal(obj):
            # If numpy array
            if isinstance(obj, np.ndarray):
                # Direct numeric array
                if obj.dtype != object:
                    return obj.flatten()
                # Object array: descend into elements
                for elem in obj.flatten():
                    sig = extract_signal(elem)
                    if sig is not None:
                        return sig
            else:
                # MATLAB struct or other object: search its attributes
                for attr in [a for a in dir(obj) if not a.startswith('_')]:
                    try:
                        val = getattr(obj, attr)
                    except Exception:
                        continue
                    sig = extract_signal(val)
                    if sig is not None:
                        return sig
            return None

        merged_dir = os.path.join(self.output_base_path, "_merged_temp")
        os.makedirs(merged_dir, exist_ok=True)

        label_map = {
            "BPFI": "BPFI",
            "BPFO": "BPFO",
            "Normal": "Normal"
        }

        # Step 1: Load .mat and extract
        for file in tqdm(os.listdir(self.raw_data_path), desc="Processing .mat files"):
            if not file.endswith(".mat"):
                continue
            file_path = os.path.join(self.raw_data_path, file)
            class_key = next((k for k in label_map if k in file), None)
            if class_key is None:
                print(f"⚠️ Skipping unrecognized file: {file}")
                continue

            class_name = label_map[class_key]
            class_dir = os.path.join(merged_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            try:
                mat = scipy.io.loadmat(file_path, struct_as_record=False, squeeze_me=False)
                wrapper = mat.get("Signal")
                if wrapper is None:
                    print(f"❌ 'Signal' key not found in {file}")
                    continue

                # Extract signal
                signal = extract_signal(wrapper)
                if signal is None:
                    print(f"❌ Could not find numeric signal in {file}")
                    continue

                # Save waveform
                out_name = os.path.splitext(file)[0] + ".npy"
                np.save(os.path.join(class_dir, out_name), signal)

            except Exception as e:
                print(f"❌ Error processing {file}: {e}")

        # Step 2: Split data
        print("\n📦 Splitting into train/val/test...")
        all_files = []
        for cls in sorted(os.listdir(merged_dir)):
            for fname in os.listdir(os.path.join(merged_dir, cls)):
                all_files.append((os.path.join(merged_dir, cls, fname), cls))

        total = len(all_files)
        train_end = int(0.7 * total)
        val_end = train_end + int(0.15 * total)
        rng = np.random.default_rng(42)
        rng.shuffle(all_files)
        splits = {'train': all_files[:train_end],
                  'val': all_files[train_end:val_end],
                  'test': all_files[val_end:]}

        for split, items in splits.items():
            for path, cls in tqdm(items, desc=f"Copying {split}"):
                dst = os.path.join(self.output_base_path, split, cls)
                os.makedirs(dst, exist_ok=True)
                shutil.copy(path, os.path.join(dst, os.path.basename(path)))

        shutil.rmtree(merged_dir)
        print(f"\n✅ Acoustic dataset ready at {self.output_base_path}")
