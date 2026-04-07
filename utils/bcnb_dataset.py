import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import random   

class BCNB_Dataset(Dataset):
    def __init__(self, txt_file_path, percentage=None, nas_stage=None):
        self.image_paths = []
        self.imclass_data_dict = {}
        with open(txt_file_path, 'r') as f:
            self.image_paths = [line.strip() for line in f.readlines() if line.strip()]

        df = pd.read_excel('data/patient-clinical-data.xlsx')  # Replace with your file path
        imID_and_Class = df[['Patient ID', 'ALN status']].values.tolist()
        self.imclass_data_dict = {item[0]: item[1] for item in imID_and_Class}

        self.class_map = {        
                        "N+(>2)": 1,
                        "N+(1-2)": 1,
                        "N0": 0
                        }
        self.transform = transforms.Compose([transforms.ToTensor()])

        # Perform sampling if percentage is provided and in nas_stage
        if percentage is not None and nas_stage == True:
            # Group image paths by class
            class_to_paths = {0: [], 1: []}
            for path in self.image_paths:
                base_name = os.path.basename(os.path.dirname(path))
                class_status = self.imclass_data_dict.get(int(base_name), None)
                if class_status in self.class_map:
                    class_id = self.class_map[class_status]
                    class_to_paths[class_id].append(path)

            # Sample percentage from each class
            random.seed(42)
            sampled_paths = []
            nbr_samples = int(len(self.image_paths) * percentage / len(class_to_paths.keys()))
            for class_id, paths in class_to_paths.items():
                n = max(1, int(nbr_samples))
                sampled_paths.extend(random.sample(paths, n) if len(paths) > 0 else [])

            self.image_paths = sampled_paths

    def __len__(self):  
        return len(self.image_paths)
    
    def __getitem__(self, idx): 
        patch_path = self.image_paths[idx]
        base_name = os.path.basename(os.path.dirname(patch_path))
        class_status = self.imclass_data_dict.get(int(base_name), None)
        class_id = self.class_map[class_status]
        img = Image.open(patch_path).convert('RGB')       
        
        if self.transform:
            img = self.transform(img)
        return  img, class_id, patch_path