import scipy, np, os

file_path = 'datasets/raw/acoustic/0Nm_Normal.mat'

try:
    mat_data = scipy.io.loadmat(file_path, struct_as_record=False, squeeze_me=True)

    signal = mat_data.get("Signal", None)

    if signal is None:
        print(f"❌ 'Signal' key not found in {file}")
        continue

    # If it's an ndarray of shape (1, N) or (N, 1), flatten it
    if isinstance(signal, np.ndarray):
        signal = signal.flatten()
    else:
        print(f"❌ Unexpected 'Signal' type in {file}: {type(signal)}")
        continue

    # Save to .npy
    base_name = os.path.splitext(file)[0]
    save_path = os.path.join(class_dir, f"{base_name}.npy")
    np.save(save_path, signal)

except Exception as e:
    print(f"❌ Failed to process {file}: {e}")