from processors.factory_maker import make_factory_objects

def main():
    print("Processing acoustic data...")
    processor = make_factory_objects("acoustic")
    processor.process()
    print("Data processing complete!")

if __name__ == "__main__":
    main() 