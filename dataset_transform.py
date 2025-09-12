import cv2
import torch
from torchvision import datasets, transforms

class BGRDataset4train(datasets.ImageFolder):
    def __init__(
        self,
        root,
        loader=None,
        is_valid_file=None,
        bluefish_class_names=None,      # ★リスト（配列）で渡す
        manual_bluefish_ids=None,       # 数値IDで直接指定も可
        match_mode="contains"           # "contains" or "exact"
    ):
        super(BGRDataset4train, self).__init__(root=root, loader=loader, is_valid_file=is_valid_file)

        # 青魚クラスIDの決定
        if manual_bluefish_ids is not None:
            self.bluefish_ids = set(map(int, manual_bluefish_ids))
        else:
            self.bluefish_ids = set()
            if bluefish_class_names:  # リストが与えられた場合のみ探索
                for name, idx in self.class_to_idx.items():
                    for key in bluefish_class_names:
                        if (match_mode == "exact" and name == key) or \
                           (match_mode == "contains" and key in name):
                            self.bluefish_ids.add(idx)
                            break  # 1命中で十分

        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)

        # 青魚専用（色Jitter弱＋微シャープネス、回転は小さめ）
        self.bluefish_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02),
            transforms.RandomAdjustSharpness(sharpness_factor=1.2, p=0.5),
            transforms.RandomHorizontalFlip(p=0.1),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        # 非青魚（元の設定を踏襲）
        self.default_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.3, contrast=0.1, saturation=0.1),
            transforms.RandomHorizontalFlip(0.1),
            transforms.RandomRotation(degrees=180),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=11, sigma=(0.1, 2.0))], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def get_num_classes(self):
        return len(self.classes)

    def save_labels_to_file(self, filename):
        with open(filename, "w", encoding="utf-8") as f:
            for label in self.classes:
                f.write(label + "\n")

    def __getitem__(self, index):
        image_path, target = self.samples[index]
        img = cv2.imread(image_path)
        if img is None:
            print(image_path)

        # ★元ソースのまま
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if int(target) in self.bluefish_ids:
            sample = self.bluefish_transform(img)
        else:
            sample = self.default_transform(img)
        return sample, target
