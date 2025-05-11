from torch.utils.data import DataLoader
from dataset import KucingAnjingDataset


train_dataset = KucingAnjingDataset(root_dir='data/catndog/train')
test_dataset  = KucingAnjingDataset(root_dir='data/catndog/test')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

