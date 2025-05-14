import sys
import importlib

COMMAND_MAP = {
    'process_data': 'commands.process_data'
}

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <command> [args...]")
        print("Available commands:", ', '.join(COMMAND_MAP.keys()))
        sys.exit(1)

    command_name = sys.argv[1]

    if command_name not in COMMAND_MAP:
        print(f"Unknown command: {command_name}")
        print("Available commands:", ', '.join(COMMAND_MAP.keys()))
        sys.exit(1)

    module_path = COMMAND_MAP[command_name]
    module = importlib.import_module(module_path)
    module.chain.run()

if __name__ == "__main__":
    main()