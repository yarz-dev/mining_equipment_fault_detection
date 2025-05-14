import os 
import shutil

class ProcessDataBase():
    raw_data_path: str
    output_base_path: str

    def process(self):
        self.clear_old_data()
        self.handle_data_processing()
    
    def handle_data_processing(self):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def clear_old_data(self):
        if os.path.exists(self.output_base_path):
            print(f"Removing existing output directory: {self.output_base_path}")
            shutil.rmtree(self.output_base_path)
        