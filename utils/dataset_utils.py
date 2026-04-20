import os
import random
import copy
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
import torch

from utils.image_utils import random_augmentation, crop_img
from utils.degradation_utils import Degradation

    
class PromptTrainDataset(Dataset):
    def __init__(self, args):
        super(PromptTrainDataset, self).__init__()
        self.args = args
        self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.legacy_train_root = os.path.join(self.repo_root, 'data', 'Train')
        self.rs_ids = []
        self.hazy_ids = []
        self.motion_blur_ids = []
        self.D = Degradation(args)
        self.de_temp = 0
        self.de_type = self.args.de_type
        self.data_split = getattr(self.args, 'data_split', 'train')
        self.degradation_size = getattr(self.args, 'degradation_size', 8192)
        if self.degradation_size is None or self.degradation_size <= 0:
            self.degradation_size = None
            self.denoise_level_size = None
        else:
            self.denoise_level_size = max(1, self.degradation_size // 2)
        print(self.de_type)
        print("Data split : {}".format(self.data_split))

        # Keep backward compatibility for "deblur" while enabling explicit motion blur training.
        self.de_dict = {
            'denoise_15': 0,
            'denoise_25': 1,
            'denoise_50': 2,
            'derain': 3,
            'dehaze': 4,
            'deblur': 5,
            'motion_blur': 5,
            'de_motion_blur': 5,
        }

        self._init_ids()
        self._merge_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])

        self.toTensor = ToTensor()

    def _build_fixed_size_ids(self, clean_ids, de_type, target_size):
        tagged_ids = [{"clean_id": x, "de_type": de_type} for x in clean_ids]
        if len(tagged_ids) == 0:
            return []

        # For val/test, keep all source ids without synthetic repetition.
        if target_size is None:
            out_ids = tagged_ids.copy()
            random.shuffle(out_ids)
            return out_ids

        if target_size <= 0:
            return []

        if len(tagged_ids) >= target_size:
            out_ids = random.sample(tagged_ids, target_size)
        else:
            repeat = (target_size + len(tagged_ids) - 1) // len(tagged_ids)
            out_ids = (tagged_ids * repeat)[:target_size]

        random.shuffle(out_ids)
        return out_ids

    def _resolve_manifest(self, sub_dir, base_name):
        split_name = f"{base_name}_{self.data_split}.txt"
        split_path = os.path.join(self.args.data_file_dir, sub_dir, split_name)
        default_path = os.path.join(self.args.data_file_dir, sub_dir, f"{base_name}.txt")
        if os.path.exists(split_path):
            return split_path
        return default_path

    def _normalize_hazy_rel_path(self, rel_path):
        rel_path = rel_path.strip().replace('\\', '/')
        if rel_path.startswith('synthetic/part1/'):
            return rel_path.replace('synthetic/part1/', 'synthetic/OTS/', 1)
        return rel_path

    def _resolve_hazy_path(self, rel_path):
        rel_path = rel_path.strip().replace('\\', '/')
        normalized_rel = self._normalize_hazy_rel_path(rel_path)

        candidate_rels = [normalized_rel]
        if normalized_rel.startswith('synthetic/OTS/'):
            candidate_rels.append(normalized_rel.replace('synthetic/OTS/', 'synthetic/', 1))
        if rel_path != normalized_rel:
            candidate_rels.append(rel_path)

        for candidate_rel in candidate_rels:
            candidate_path = os.path.join(self.args.dehaze_dir, candidate_rel)
            if os.path.exists(candidate_path):
                return candidate_path

        # Fallback to legacy PromptIR layout when dataset/haze is not extracted yet.
        legacy_candidates = []
        for candidate_rel in candidate_rels:
            mapped_rel = candidate_rel
            if candidate_rel.startswith('haze/reside_ots/haze/'):
                mapped_rel = candidate_rel.replace('haze/reside_ots/haze/', 'synthetic/OTS/', 1)
            legacy_candidates.append(os.path.join(self.legacy_train_root, 'Dehaze', mapped_rel))

        for candidate_path in legacy_candidates:
            if os.path.exists(candidate_path):
                return candidate_path

        # Keep deterministic behavior even when files are not present yet.
        return os.path.join(self.args.dehaze_dir, normalized_rel)

    def _init_ids(self):
        if 'denoise_15' in self.de_type or 'denoise_25' in self.de_type or 'denoise_50' in self.de_type:
            self._init_clean_ids()
        if 'derain' in self.de_type:
            self._init_rs_ids()
        if 'dehaze' in self.de_type:
            self._init_hazy_ids()
        if self._is_motion_blur_enabled():
            self._init_motion_blur_ids()

        random.shuffle(self.de_type)

    def _is_motion_blur_enabled(self):
        return (
            'motion_blur' in self.de_type
            or 'de_motion_blur' in self.de_type
            or 'deblur' in self.de_type
        )

    def _init_clean_ids(self):
        ref_file = self._resolve_manifest("noisy", "denoise_airnet")
        temp_ids = []
        temp_ids+= [id_.strip() for id_ in open(ref_file)]
        temp_id_set = set(temp_ids)
        temp_id_basename_set = {os.path.basename(x) for x in temp_ids}

        denoise_dir = self.args.denoise_dir
        if not os.path.isdir(denoise_dir):
            fallback_denoise_dir = os.path.join(self.legacy_train_root, 'Denoise')
            if os.path.isdir(fallback_denoise_dir):
                denoise_dir = fallback_denoise_dir + '/'
                print("[Info] denoise_dir not found, fallback to {}".format(denoise_dir))

        clean_ids = []
        name_list = os.listdir(denoise_dir)
        clean_ids += [
            denoise_dir + id_
            for id_ in name_list
            if id_.strip() in temp_id_set or id_.strip() in temp_id_basename_set
        ]

        if 'denoise_15' in self.de_type:
            self.s15_ids = self._build_fixed_size_ids(clean_ids, de_type=0, target_size=self.denoise_level_size)
            self.s15_counter = 0
        if 'denoise_25' in self.de_type:
            self.s25_ids = self._build_fixed_size_ids(clean_ids, de_type=1, target_size=self.denoise_level_size)
            self.s25_counter = 0
        if 'denoise_50' in self.de_type:
            self.s50_ids = self._build_fixed_size_ids(clean_ids, de_type=2, target_size=self.denoise_level_size)
            self.s50_counter = 0

        self.num_clean = len(clean_ids)
        print("Total Denoise Source Ids : {}".format(self.num_clean))
        if self.denoise_level_size is None:
            print("Target Denoise Size Per Level : full split")
        else:
            print("Target Denoise Size Per Level : {}".format(self.denoise_level_size))

    def _init_hazy_ids(self):
        temp_ids = []
        hazy = self._resolve_manifest("hazy", "hazy_outside")
        temp_ids += [self._resolve_hazy_path(id_.strip()) for id_ in open(hazy)]
        self.hazy_ids = self._build_fixed_size_ids(temp_ids, de_type=4, target_size=self.degradation_size)

        self.hazy_counter = 0
        
        self.num_hazy = len(self.hazy_ids)
        print("Total Hazy Ids : {}".format(self.num_hazy))

    def _init_rs_ids(self):
        temp_ids = []
        rs = self._resolve_manifest("rainy", "rainTrain")
        temp_ids+= [self.args.derain_dir + id_.strip() for id_ in open(rs)]
        self.rs_ids = self._build_fixed_size_ids(temp_ids, de_type=3, target_size=self.degradation_size)

        self.rl_counter = 0
        self.num_rl = len(self.rs_ids)
        print("Total Rainy Ids : {}".format(self.num_rl))

    def _init_motion_blur_ids(self):
        motion_blur_root = getattr(
            self.args,
            "motion_blur_dir",
            "/home/huhao/adv_ir/dataset/motion_sim",
        )
        if not os.path.isabs(motion_blur_root):
            motion_blur_root = os.path.abspath(os.path.join(self.repo_root, motion_blur_root))

        split_root = os.path.join(motion_blur_root, self.data_split)
        input_dir = os.path.join(split_root, "input")
        target_dir = os.path.join(split_root, "target")

        # Support both split layout (<root>/<split>/input|target) and direct layout (<root>/input|target).
        if not (os.path.isdir(input_dir) and os.path.isdir(target_dir)):
            input_dir = os.path.join(motion_blur_root, "input")
            target_dir = os.path.join(motion_blur_root, "target")

        if not (os.path.isdir(input_dir) and os.path.isdir(target_dir)):
            raise FileNotFoundError(
                "motion_blur dataset not found, expected input/target dirs under "
                f"{os.path.join(motion_blur_root, self.data_split)} or {motion_blur_root}"
            )

        valid_exts = (".png", ".jpg", ".jpeg", ".bmp")
        input_names = sorted(
            name for name in os.listdir(input_dir)
            if name.lower().endswith(valid_exts)
        )

        paired_ids = []
        for name in input_names:
            input_path = os.path.join(input_dir, name)
            target_path = os.path.join(target_dir, name)
            if os.path.exists(target_path):
                paired_ids.append(
                    {
                        "clean_id": input_path,
                        "target_id": target_path,
                        "de_type": 5,
                    }
                )

        if len(paired_ids) == 0:
            raise RuntimeError(
                f"motion_blur dataset has no valid input-target pairs: {input_dir} <-> {target_dir}"
            )

        if self.degradation_size is None:
            self.motion_blur_ids = paired_ids.copy()
            random.shuffle(self.motion_blur_ids)
        elif len(paired_ids) >= self.degradation_size:
            self.motion_blur_ids = random.sample(paired_ids, self.degradation_size)
        else:
            repeat = (self.degradation_size + len(paired_ids) - 1) // len(paired_ids)
            self.motion_blur_ids = (paired_ids * repeat)[:self.degradation_size]
            random.shuffle(self.motion_blur_ids)

        self.motion_blur_counter = 0
        self.num_motion_blur = len(self.motion_blur_ids)
        print("Total Motion Blur Source Pairs : {}".format(len(paired_ids)))
        print("Total Motion Blur Ids : {}".format(self.num_motion_blur))
    

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def _center_crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = max(0, (H - self.args.patch_size) // 2)
        ind_W = max(0, (W - self.args.patch_size) // 2)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def _center_crop_single(self, img):
        H = img.shape[0]
        W = img.shape[1]
        ind_H = max(0, (H - self.args.patch_size) // 2)
        ind_W = max(0, (W - self.args.patch_size) // 2)
        return img[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

    def _get_gt_name(self, rainy_name):
        # Original PromptIR layout: .../rainy/rain-xxx.png -> .../gt/norain-xxx.png
        if "rainy/" in rainy_name and "rain-" in rainy_name:
            return rainy_name.split("rainy")[0] + 'gt/norain-' + rainy_name.split('rain-')[-1]

        # rain13K layout: .../input/xxxx.jpg -> .../target/xxxx.jpg
        if "/input/" in rainy_name:
            return rainy_name.replace('/input/', '/target/')
        if "input/" in rainy_name:
            return rainy_name.replace('input/', 'target/')

        raise ValueError(f"Unsupported derain path format: {rainy_name}")

    def _get_nonhazy_name(self, hazy_name):
        # New dataset layout: .../haze/reside_ots/haze/xxxx_a_b.jpg -> .../haze/reside_ots/clear/xxxx.jpg
        if '/haze/reside_ots/haze/' in hazy_name:
            clear_name = hazy_name.replace('/haze/reside_ots/haze/', '/haze/reside_ots/clear/')
            base_name = os.path.basename(clear_name)
            stem, ext = os.path.splitext(base_name)
            clean_stem = stem.split('_')[0]
            dataset_clear = os.path.join(os.path.dirname(clear_name), clean_stem + ext)
            if os.path.exists(dataset_clear):
                return dataset_clear

            # Fallback for legacy dehaze root.
            return os.path.join(self.legacy_train_root, 'Dehaze', 'original', clean_stem + ext)

        dir_name = hazy_name.split("synthetic")[0] + 'original/'
        name = hazy_name.split('/')[-1].split('_')[0]
        suffix = '.' + hazy_name.split('.')[-1]
        nonhazy_name = dir_name + name + suffix
        return nonhazy_name

    def _merge_ids(self):
        self.sample_ids = []
        if "denoise_15" in self.de_type:
            self.sample_ids += self.s15_ids
        if "denoise_25" in self.de_type:
            self.sample_ids += self.s25_ids
        if "denoise_50" in self.de_type:
            self.sample_ids += self.s50_ids
        if "derain" in self.de_type:
            self.sample_ids+= self.rs_ids
        
        if "dehaze" in self.de_type:
            self.sample_ids+= self.hazy_ids
        if self._is_motion_blur_enabled():
            self.sample_ids += self.motion_blur_ids
        print(len(self.sample_ids))

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]

        if de_id < 3:
            if de_id == 0:
                clean_id = sample["clean_id"]
            elif de_id == 1:
                clean_id = sample["clean_id"]
            elif de_id == 2:
                clean_id = sample["clean_id"]

            clean_img = crop_img(np.array(Image.open(clean_id).convert('RGB')), base=16)
            if self.data_split == "train":
                clean_patch = self.crop_transform(clean_img)
                clean_patch = np.array(clean_patch)
            else:
                clean_patch = self._center_crop_single(clean_img)

            clean_name = clean_id.split("/")[-1].split('.')[0]

            if self.data_split == "train":
                clean_patch = random_augmentation(clean_patch)[0]

            degrad_patch = self.D.single_degrade(clean_patch, de_id)
        else:
            if de_id == 3:
                # Rain Streak Removal
                degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
                clean_name = self._get_gt_name(sample["clean_id"])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            elif de_id == 4:
                # Dehazing with SOTS outdoor training set
                degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
                clean_name = self._get_nonhazy_name(sample["clean_id"])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            elif de_id == 5:
                # Motion blur paired dataset: input/<name> <-> target/<name>
                degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
                clean_name = os.path.splitext(os.path.basename(sample["target_id"]))[0]
                clean_img = crop_img(np.array(Image.open(sample["target_id"]).convert('RGB')), base=16)
            else:
                raise ValueError(f"Unsupported de_type id: {de_id}")

            if self.data_split == "train":
                degrad_patch, clean_patch = random_augmentation(*self._crop_patch(degrad_img, clean_img))
            else:
                degrad_patch, clean_patch = self._center_crop_patch(degrad_img, clean_img)

        clean_patch = self.toTensor(clean_patch)
        degrad_patch = self.toTensor(degrad_patch)


        return [clean_name, de_id], degrad_patch, clean_patch

    def __len__(self):
        return len(self.sample_ids)


class DenoiseTestDataset(Dataset):
    def __init__(self, args):
        super(DenoiseTestDataset, self).__init__()
        self.args = args
        self.clean_ids = []
        self.sigma = 15

        self._init_clean_ids()

        self.toTensor = ToTensor()

    def _init_clean_ids(self):
        name_list = os.listdir(self.args.denoise_path)
        self.clean_ids += [self.args.denoise_path + id_ for id_ in name_list]

        self.num_clean = len(self.clean_ids)

    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def set_sigma(self, sigma):
        self.sigma = sigma

    def __getitem__(self, clean_id):
        clean_img = crop_img(np.array(Image.open(self.clean_ids[clean_id]).convert('RGB')), base=16)
        clean_name = self.clean_ids[clean_id].split("/")[-1].split('.')[0]

        noisy_img, _ = self._add_gaussian_noise(clean_img)
        clean_img, noisy_img = self.toTensor(clean_img), self.toTensor(noisy_img)

        return [clean_name], noisy_img, clean_img
    def tile_degrad(input_,tile=128,tile_overlap =0):
        sigma_dict = {0:0,1:15,2:25,3:50}
        b, c, h, w = input_.shape
        tile = min(tile, h, w)
        assert tile % 8 == 0, "tile size should be multiple of 8"

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h, w).type_as(input_)
        W = torch.zeros_like(E)
        s = 0
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = input_[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = in_patch
                # out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(in_patch)

                E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
                W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
        # restored = E.div_(W)

        restored = torch.clamp(restored, 0, 1)
        return restored
    def __len__(self):
        return self.num_clean


class DerainDehazeDataset(Dataset):
    def __init__(self, args, task="derain",addnoise = False,sigma = None):
        super(DerainDehazeDataset, self).__init__()
        self.ids = []
        self.task_idx = 0
        self.args = args

        self.task_dict = {'derain': 0, 'dehaze': 1}
        self.toTensor = ToTensor()
        self.addnoise = addnoise
        self.sigma = sigma

        self.set_dataset(task)
    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def _init_input_ids(self):
        if self.task_idx == 0:
            self.ids = []
            name_list = os.listdir(self.args.derain_path + 'input/')
            # print(name_list)
            print(self.args.derain_path)
            self.ids += [self.args.derain_path + 'input/' + id_ for id_ in name_list]
        elif self.task_idx == 1:
            self.ids = []
            name_list = os.listdir(self.args.dehaze_path + 'input/')
            self.ids += [self.args.dehaze_path + 'input/' + id_ for id_ in name_list]

        self.length = len(self.ids)

    def _get_gt_path(self, degraded_name):
        if self.task_idx == 0:
            gt_name = degraded_name.replace("input", "target")
        elif self.task_idx == 1:
            dir_name = degraded_name.split("input")[0] + 'target/'
            name = degraded_name.split('/')[-1].split('_')[0] + '.png'
            gt_name = dir_name + name
        return gt_name

    def set_dataset(self, task):
        self.task_idx = self.task_dict[task]
        self._init_input_ids()

    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path = self._get_gt_path(degraded_path)

        degraded_img = crop_img(np.array(Image.open(degraded_path).convert('RGB')), base=16)
        if self.addnoise:
            degraded_img,_ = self._add_gaussian_noise(degraded_img)
        clean_img = crop_img(np.array(Image.open(clean_path).convert('RGB')), base=16)

        clean_img, degraded_img = self.toTensor(clean_img), self.toTensor(degraded_img)
        degraded_name = degraded_path.split('/')[-1][:-4]

        return [degraded_name], degraded_img, clean_img

    def __len__(self):
        return self.length


class TestSpecificDataset(Dataset):
    def __init__(self, args):
        super(TestSpecificDataset, self).__init__()
        self.args = args
        self.degraded_ids = []
        self._init_clean_ids(args.test_path)

        self.toTensor = ToTensor()

    def _init_clean_ids(self, root):
        extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']
        if os.path.isdir(root):
            name_list = []
            for image_file in os.listdir(root):
                if any([image_file.endswith(ext) for ext in extensions]):
                    name_list.append(image_file)
            if len(name_list) == 0:
                raise Exception('The input directory does not contain any image files')
            self.degraded_ids += [root + id_ for id_ in name_list]
        else:
            if any([root.endswith(ext) for ext in extensions]):
                name_list = [root]
            else:
                raise Exception('Please pass an Image file')
            self.degraded_ids = name_list
        print("Total Images : {}".format(name_list))

        self.num_img = len(self.degraded_ids)

    def __getitem__(self, idx):
        degraded_img = crop_img(np.array(Image.open(self.degraded_ids[idx]).convert('RGB')), base=16)
        name = self.degraded_ids[idx].split('/')[-1][:-4]

        degraded_img = self.toTensor(degraded_img)

        return [name], degraded_img

    def __len__(self):
        return self.num_img
    
