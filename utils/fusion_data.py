import cv2
import torch
from kornia.utils import image_to_tensor
from pathlib import Path
from torch import Tensor
from torch.utils.data import Dataset


class FusionData(Dataset):
    """
    Loading fusion data from hard disk.
    """

    def __init__(self, folder: Path, mask: str = 'm1', mode='train', use_data='custom', use_model='TarDAL', transforms=lambda x: x):
        super(FusionData, self).__init__()

        assert mode in ['eval', 'train'], 'mode should be "eval" or "train"'
        assert use_data in ['default', 'custom'], 'use_data should be "defualt" or "custom"'
        names = (folder / 'list.txt').read_text().splitlines()
        assert len(names) > 0, 'list.txt is empty'
        ext = ('bmp', 'bmp') if use_data == 'default' else ('png', 'png')
        self.samples = [{
            'name': name,
            'ir': folder / 'ir_en' / f'{name}.{ext[0]}',
            'vi': folder / 'vi' / f'{name}.{ext[1]}',
            'mk': folder / 'mask' / mask / f'{name}.png',
            'vsm': {
                'ir': folder / 'vsm' / 's1' / f'{name}.bmp',
                'vi': folder / 'vsm' / 's2' / f'{name}.bmp'
            },
        } for name in names]
        self.transforms = transforms
        self.mode = mode
        self.use_model = use_model

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        ir, vi = self.imread(sample['ir']), self.imread(sample['vi'])
        if self.mode == 'train' and self.use_model == 'TarDAL':
            mk = self.imread(sample['mk'])
            s1, s2 = self.imread(sample['vsm']['ir']), self.imread(sample['vsm']['vi'])
            im = torch.cat([ir, vi, mk, s1, s2], dim=0)
            im = self.transforms(im)
            ir, vi, mk, s1, s2 = torch.chunk(im, 5, dim=0)
            sample = {'name': sample['name'], 'ir': ir, 'vi': vi, 'mk': mk, 'vsm': {'ir': s1, 'vi': s2}}
        if self.mode == 'train' and self.use_model == 'DDcGAN':
            mk = torch.zeros_like(ir)
            s1, s2 = torch.zeros_like(ir), torch.zeros_like(vi)
            im = torch.cat([ir, vi, mk, s1, s2], dim=0)
            im = self.transforms(im)
            ir, vi, mk, s1, s2 = torch.chunk(im, 5, dim=0)
            sample = {'name': sample['name'], 'ir': ir, 'vi': vi, 'mk': mk, 'vsm': {'ir': s1, 'vi': s2}}
        elif self.mode == 'eval':
            im = torch.cat([ir, vi], dim=0)
            im = self.transforms(im)
            ir, vi = torch.chunk(im, 2, dim=0)
            sample = {'name': sample['name'], 'ir': ir, 'vi': vi}
        return sample

    @staticmethod
    def imread(path: Path) -> Tensor:
        img_n = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        img_t = image_to_tensor(img_n / 255.).float()
        return img_t


if __name__ == '__main__':
    fd = FusionData(folder=Path('../data/train/irfissure'), use_data='custom', use_model='DDcGAN', mode='train')
    s = fd[0]
    print(s)
