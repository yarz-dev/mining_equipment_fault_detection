# Mining Equipment Fault Detection

This project is designed to detect faults in mining equipment using various datasets, including thermal, acoustic, vibration, and temperature data. The system processes raw datasets, cleans them, and applies fault detection algorithms.

## Project Structure

```plaintext
├── commands/
│   ├── __init__.py
│   ├── base.py
│   ├── process_data.py
├── datasets/
│   ├── cleaned/
│   │   ├── thermal/
│   │   ├── ...
│   ├── raw/
│   │   ├── thermal/
│   │   ├── acoustic/
│   │   ├── ...
├── processors/
│   ├── __init__.py
│   ├── base.py
│   ├── factory_maker.py
│   ├── thermal.py
│   ├── types.py
├── config.py
├── main.py
├── requirements.txt
├── README.md
├── LICENSE
```

### Key Directories

- **`commands/`**: Contains command modules, such as `process_data.py`, which handle specific tasks.
- **`datasets/`**: Stores raw and cleaned datasets organized by type (e.g., thermal, acoustic).
- **`processors/`**: Includes modules for processing datasets and implementing fault detection logic.

## Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/your-repo/mining_equipment_fault_detection.git
   cd mining_equipment_fault_detection
   ```

2. Install dependencies:

   ```sh
   pip install -r requirements.txt
   ```

## Usage

Run the main script with a command:

```sh
python main.py <command> [args...]
```

### Available Commands

- **`process_data`**: Processes raw datasets and prepares them for analysis.

Example:

```sh
python main.py process_data
```

## Configuration

Dataset paths are configured in `config.py`. You can modify the paths to suit your environment.

### Example Paths

- Raw thermal dataset: `datasets/raw/thermal/`
- Cleaned thermal dataset: `datasets/cleaned/thermal/`

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

For questions or support, please contact the project maintainer.
