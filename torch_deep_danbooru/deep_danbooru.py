import os
import re
from PIL import Image
import requests
import numpy as np
import torch
from torch import autocast
from .deep_danbooru_model import DeepDanbooruModel

re_special = re.compile(r'([\\()])')

class DeepDanbooru:
    def __init__(self, model_path, half=True, gpu_id=0, image_size=512):
        try:
            self.model_path = model_path
            self.half = half
            self.gpu_id = gpu_id
            self.image_size = image_size

            if model_path is None:
                raise ValueError('model_path is None')
            if os.path.exists(self.model_path):
                print(f'DeepDanbooru: Loading model from {self.model_path}')
            else:
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                url = 'https://github.com/AUTOMATIC1111/TorchDeepDanbooru/releases/download/v1/model-resnet_custom_v3.pt'
                print(f'DeepDanbooru: Downloading model {self.model_path}')
                r = requests.get(url, allow_redirects=True)
                open(self.model_path, 'wb').write(r.content)


            self.device = torch.device('cuda')
            print(f'DeepDanbooru: Using device {self.device}')
            self.model = DeepDanbooruModel()
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))

            self.model.eval()
            if self.half:
                self.model.half()
            self.model = self.model.to(self.device)
            print(f'DeepDanbooru: Model loaded')
        except Exception as e:
            print(f'Initiate DeepDanbooru failed: {e}')
            raise e

    def get_tag(self, pil_image, threshold):
        try:
            threshold = threshold
            use_spaces = True
            use_escape = True
            alpha_sort = True
            include_ranks = False

            pic = resize_image(pil_image.convert("RGB"), 512, 512)
            a = np.expand_dims(np.array(pic, dtype=np.float32), 0) / 255

            with torch.no_grad(), autocast("cuda"):
                x = torch.from_numpy(a).to(self.device)
                y = self.model(x).cpu().numpy()[0]
            torch.cuda.empty_cache()

            probability_dict = {}

            print(f'DeepDanbooru: Threshold: {threshold:.3f}')
            for tag, probability in zip(self.model.tags, y):
                if tag.startswith("rating:") or tag.startswith("black_border") or tag.startswith("letterboxed") or tag.startswith("pillarboxed") or tag.startswith("tokyo_(city)"):
                    continue
                elif probability < threshold:
                    print(f'DeepDanbooru: Possible tag (-): {tag}: {probability:.3f}')
                    continue
                else:
                    print(f'DeepDanbooru: Possible tag (+): {tag}: {probability:.3f}')
                    probability_dict[tag] = probability

            if alpha_sort:
                tags = sorted(probability_dict)
            else:
                tags = [tag for tag, _ in sorted(probability_dict.items(), key=lambda x: -x[1])]

            res = []

            for tag in tags:
                probability = probability_dict[tag]
                tag_outformat = tag
                if use_spaces:
                    tag_outformat = tag_outformat.replace('_', ' ')
                if use_escape:
                    tag_outformat = re.sub(re_special, r'\\\1', tag_outformat)
                if include_ranks:
                    tag_outformat = f"({tag_outformat}:{probability:.3f})"

                res.append(tag_outformat)

            return ", ".join(res)
        except Exception as e:
            torch.cuda.empty_cache()
            print(f'DeepDanbooru: Error: {e}')
            return ''
            


def resize_image(im, width, height):
    ratio = width / height
    src_ratio = im.width / im.height

    src_w = width if ratio < src_ratio else im.width * height // im.height
    src_h = height if ratio >= src_ratio else im.height * width // im.width

    resized = im.resize((src_w, src_h), Image.LANCZOS)
    res = Image.new("RGB", (width, height))
    res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

    if ratio < src_ratio:
        fill_height = height // 2 - src_h // 2
        res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
        res.paste(resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)), box=(0, fill_height + src_h))
    elif ratio > src_ratio:
        fill_width = width // 2 - src_w // 2
        res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
        res.paste(resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)), box=(fill_width + src_w, 0))

    return res