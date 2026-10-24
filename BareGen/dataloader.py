from .import_packages import *
from .utils import *

resolution_pool = [
    # ── 1:1 正方形 ──
    (128, 128), (256, 256), (384, 384), (512, 512),
    (640, 640), (768, 768), (896, 896), (1024, 1024),
    # ── 4:3 / 3:4 ──
    (384, 512), (512, 384),
    (480, 640), (640, 480),
    (576, 768), (768, 576),
    (624, 832), (832, 624),
    (768, 1024), (1024, 768),
    # ── 16:9 / 9:16 ──
    (288, 512), (512, 288),
    (368, 640), (640, 368),
    (432, 768), (768, 432),
    (504, 896), (896, 504),
    (576, 1024), (1024, 576),

    # ── 3:2 / 2:3 ──
    (320, 480), (480, 320),
    (512, 768), (768, 512),
    (640, 960), (960, 640),

    # ── 2:1 / 1:2 ──
    (256, 512), (512, 256),
    (384, 768), (768, 384),
    (512, 1024), (1024, 512),
]
batch_normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def get_transform_batch_diff_resolution():
    index = torch.randint(low=0, high=len(resolution_pool), size=()).item()
    size = resolution_pool[index]
    transform = transforms.Compose([
        transforms.RandomCrop(size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform


def collect_fn_batch_diff_resolution(batch):
    transform = get_transform_batch_diff_resolution()
    return torch.stack([transform(item) for item in batch], dim=0)


jpeg_noise = kornia.augmentation.RandomJPEG(
    jpeg_quality=(30, 95),
    same_on_batch=False,
    p=0.5,
)


def preprocessing_add_noise(data):
    if torch.rand(1).item() < 0.5:
        noise = torch.randn_like(data) * random.uniform(0.005, 0.01)
        lr = data + noise
    else:
        poisson_rate = data * random.uniform(2.0, 5.0)
        poisson_rate = torch.clamp(poisson_rate, min=1e-6)
        poisson_noise = torch.poisson(poisson_rate) / 10.0
        lr = data + poisson_noise
    lr = torch.clamp(lr, 0.0, 1.0)
    if torch.rand(1).item() < 0.35:
        lr = jpeg_noise(lr)
    return lr


def collate_fn_add_noise(batch):
    hr = []
    lr = []
    for item in batch:
        hr_ = item[1]
        lr_ = preprocessing_add_noise(item[0])
        hr_ = batch_normalize(hr_)
        lr_ = batch_normalize(lr_)
        hr.append(hr_)
        lr.append(lr_)
    hr = torch.stack(hr, dim=0)
    lr = torch.stack(lr, dim=0)
    return lr, hr


def default_transform(transform=None):
    if transform is None:
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    return transform


class ImageDataset(Dataset):
    def __init__(self, root_path, transform=None, to_tensor=True):
        self.to_tensor = to_tensor
        self.root_path = root_path
        self.transform = default_transform(transform)
        self.image_paths = get_all_file_paths(root_path=root_path, ends_with=("jpg", "jpeg", "png"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index]).convert('RGB')
        if self.to_tensor:
            img = self.transform(img)
            return img
        return img


class SuperResolutionDataset(Dataset):
    """
    get_dataloader(dataset,
                   batch_size=16,
                   shuffle=True,
                   num_workers=4,
                   pin_memory=True,
                   persistent_workers=True,
                   collect_fn=collate_fn_add_noise)
    """

    def __init__(
            self,
            root_path,
            down_scale=4,
            crop_size=(128, 128),
    ):
        self.images_path = get_all_file_paths(root_path=root_path, ends_with=("jpg", "png", "jpeg"))
        self.down_scale = down_scale
        self.crop_size = crop_size
        self.random_crop = transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
        ])
        self.resize = transforms.Resize(
            (crop_size[0] // down_scale, crop_size[1] // down_scale),
            interpolation=transforms.InterpolationMode.BICUBIC
        )
        self.blur = kornia.augmentation.RandomGaussianBlur(
            kernel_size=(7, 7),
            sigma=(0.2, 3.0),
            p=0.8,
        )

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        hr_img = Image.open(self.images_path[index]).convert("RGB")
        hr = self.random_crop(hr_img)
        with torch.no_grad():
            lr = hr.unsqueeze(0)
            lr = self.blur(lr)
            lr = self.resize(lr)
            lr = lr.squeeze(0)
        return lr, hr


"""
不同批次分辨率不同:
class ImageDataset(IterableDataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __iter__(self):
        for data in self.dataset:
            yield data["jpg"]
get_dataloader(dataset,
                   batch_size=16,
                   shuffle=False,
                   num_workers=4,
                   pin_memory=True,
                   persistent_workers=True,
                   collect_fn=collect_fn_batch_diff_resolution)
"""


def get_dataloader(dataset,
                   batch_size=32,
                   shuffle=True,
                   num_workers=4,
                   pin_memory=True,
                   persistent_workers=True,
                   collect_fn=None):
    if collect_fn is None:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            collate_fn=collect_fn
        )
    return dataloader
