from .pdf_tools import render_pdf_to_image, render_pdf_to_text
from .destroy import random_destroy_image
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import json


class PDFDataset(Dataset):
    def __init__(
        self, 
        pdf_folder, 
        image_hw=(512, 512),
    ):
        self.pdfs = list(Path(pdf_folder).glob("**/*.pdf"))
        self.image_hw = image_hw

    def __len__(self):
        return len(self.pdfs)

    def __getitem__(self, idx):
        pdf_path = self.pdfs[idx]
        h, w = self.image_hw
        hr_image = render_pdf_to_image(pdf_path, (h*4, w*4))
        hr_image = hr_image.resize((w, h), resample=Image.LANCZOS)
        lr_image = random_destroy_image(hr_image)
        text = render_pdf_to_text(pdf_path)
        return {
            "pdf_path": str(pdf_path), 
            "lr_image": lr_image, 
            "hr_image": hr_image, 
            "text": text
        }

    def collate_fn(self, batch):
        assert len(batch) > 0, "Batch must not be empty"
        return {key: [sample[key] for sample in batch] for key in batch[0].keys()}


class ImageDataset(Dataset):
    def __init__(self, metadata_file, image_hw=(512, 512)):
        self.image_hw = image_hw
        metadata_path = Path(metadata_file)
        image_folder = metadata_path.parent
        split = 'test' if 'test' in metadata_path.name else 'train'
        self.items = []
        with open(metadata_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                file_name = data['file_name']
                gt = json.loads(data['ground_truth'])
                text = gt['gt_parse']['text_sequence']
                hr_path = image_folder / 'hr_image' / split / file_name
                lr_path = image_folder / 'lr_image' / split / file_name
                self.items.append({
                    'file_name': file_name,
                    'hr_path': hr_path,
                    'lr_path': lr_path,
                    'text': text
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        hr_image = Image.open(item['hr_path']).convert('RGB')
        lr_image = Image.open(item['lr_path']).convert('RGB')
        h, w = self.image_hw
        hr_image = hr_image.resize((w, h), resample=Image.LANCZOS)
        lr_image = lr_image.resize((w, h), resample=Image.LANCZOS)
        text = item['text']
        tag = item['file_name']
        return {
            "lr_image": lr_image,
            "hr_image": hr_image,
            "text": text,
            "tag": tag
        }

    def collate_fn(self, batch):
        assert len(batch) > 0, "Batch must not be empty"
        return {key: [sample[key] for sample in batch] for key in batch[0].keys()}
