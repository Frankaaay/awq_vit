import torch
from datasets import load_dataset


def get_calib_dataset(
    data="pileval",
    tokenizer=None,
    n_samples=512,
    block_size=512,
    image_dataset_name=None,
    image_processor=None,
    batch_size=32,
    image_split="train",
):
    if data == "pileval":
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
        dataset = dataset.shuffle(seed=42)
        samples = []
        n_run = 0
        for data in dataset:
            line = data["text"]
            line = line.strip()
            line_encoded = tokenizer.encode(line)
            if len(line_encoded) > 512:
                continue
            sample = torch.tensor([line_encoded])
            if sample.numel() == 0:
                continue
            samples.append(sample)
            n_run += 1
            if n_run == n_samples:
                break
        # now concatenate all samples and split according to block size
        cat_samples = torch.cat(samples, dim=1)
        n_split = cat_samples.shape[1] // block_size
        print(f" * Split into {n_split} blocks")
        return [
            cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
        ]
    elif data == "imagefolder":
        # New mode for image-based calibration (ViT, etc.) using HuggingFace datasets
        if image_dataset_name is None or image_processor is None:
            raise ValueError("image_dataset_name and image_processor must be provided for imagefolder mode.")
        dataset = load_dataset(image_dataset_name, split=image_split)
        dataset = dataset.shuffle(seed=42)
        image_batches = []
        n_run = 0
        for example in dataset:
            # Try to find the image key (common: 'image')
            img = example.get('image', None)
            if img is None:
                # Try other common keys
                for k in example:
                    if hasattr(example[k], 'convert'):
                        img = example[k]
                        break
            if img is None:
                continue
            processed = image_processor(img, return_tensors="pt")
            image_batches.append(processed['pixel_values'])
            n_run += 1
            if n_run == n_samples:
                break
        all_images = torch.cat(image_batches, dim=0)
        n_split = all_images.shape[0] // batch_size
        print(f" * Loaded {all_images.shape[0]} images, split into {n_split} batches")
        return [all_images[i * batch_size : (i + 1) * batch_size] for i in range(n_split)]
    else:
        raise NotImplementedError

