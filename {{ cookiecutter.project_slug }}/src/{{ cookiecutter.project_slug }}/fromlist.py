from torch.utils.data import Dataset

class DatasetFromList ( Dataset ):
    def __init__(self, texts):
        super().__init__()
        self.texts = texts
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        return self.texts[index]