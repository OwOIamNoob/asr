
from torch.utils.data import Dataset
from torchvision.transform import Transform
from PIL import Image
from mnist import MNIST

class CustomDataset(Dataset):

    def __init__(self,
                 data_dir: str,  
                 transforms: Transform):
        self.dataset = MNIST(data_dir)
        self.
        self.image_paths = image_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transforms(image)
        return {"input": image}