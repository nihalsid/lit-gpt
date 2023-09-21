import random
import torch
import numpy as np
import trimesh
from torch.utils.data import Dataset
from pathlib import Path
import torch.utils.data
import pickle
from numpy import random

from scripts.ngon_helpers import normalize_vertices, quantize_soup, scale_vertices, ngonsoup_sequence_to_mesh, plot_vertices_and_faces, shift_vertices


class NgonSoup(Dataset):

    def __init__(self, tokenizer, config, split, scale_augment, shift_augment):
        super().__init__()
        data_path = Path(config.dataset_root)
        self.tokenizer = tokenizer
        self.block_size = config.block_size
        self.cached_vertices = []
        self.cached_faces = []
        self.names = []
        self.num_tokens = config.num_tokens
        self.max_vertices = config.max_vertices
        self.max_faces = config.max_faces
        self.scale_augment = scale_augment
        self.shift_augment = shift_augment
        with open(data_path, 'rb') as fptr:
            data = pickle.load(fptr)
            if not config.overfit:
                self.names = data[f'name_{split}']
                self.cached_vertices = data[f'vertices_{split}']
                self.cached_faces = data[f'faces_{split}']
            else:
                overfit_sample_index = data[f'name_train'].index('03001627_ff2223a085d32243696b74614952b2d0_dec05')
                multiplier = 200 if split == 'val' else 20000
                self.names = data[f'name_train'][overfit_sample_index: overfit_sample_index + 1] * multiplier
                self.cached_vertices = data[f'vertices_train'][overfit_sample_index: overfit_sample_index + 1] * multiplier
                self.cached_faces = data[f'faces_train'][overfit_sample_index: overfit_sample_index + 1] * multiplier

        if config.only_chairs:
            self.cached_vertices = [self.cached_vertices[i] for i in range(len(self.cached_vertices)) if self.names[i].split('_')[0] == '03001627']
            self.cached_faces = [self.cached_faces[i] for i in range(len(self.cached_faces)) if self.names[i].split('_')[0] == '03001627']
            self.names = [self.names[i] for i in range(len(self.names)) if self.names[i].split('_')[0] == '03001627']

        print(len(self.cached_vertices), "meshes loaded")

        max_inner_face_len = 0
        for i in range(len(self.cached_vertices)):
            self.cached_vertices[i] = np.array(self.cached_vertices[i])
            for j in range(len(self.cached_faces[i])):
                max_inner_face_len = max(max_inner_face_len, len(self.cached_faces[i][j]))
        print('Longest inner face sequence', max_inner_face_len)

        self.vocab_size = config.num_tokens
        self.new_face_token = 0
        self.stop_face_token = 1
        self.pad_face_token_in = 0
        self.pad_face_token_tgt = 0
        self.padding = int(config.padding * self.block_size)

    def __getitem__(self, index: int):
        vertices = self.cached_vertices[index]
        faces = self.cached_faces[index]

        if self.scale_augment:
            vertices = scale_vertices(vertices)
        vertices = normalize_vertices(vertices)
        if self.shift_augment:
            vertices = shift_vertices(vertices)

        soup_sequence = \
            quantize_soup(vertices, faces, num_tokens=self.num_tokens - 3, max_vertices=self.max_vertices, max_faces=self.max_faces)  # throws error if processing fails
        # vertices_q = self.pad_vertices(vertices_q)

        example_text = " ".join(map(str, soup_sequence.tolist()))
        example = self.tokenizer.encode(example_text).tolist()
        example.append(self.tokenizer.eos_id)

        j = random.choice(list(range(max(1, len(example) - self.block_size + self.padding))))

        # face sequence in block format
        end_index = min(j + self.block_size, len(example))
        x_in = np.array(example[j:end_index])
        y_in = np.array(example[j:end_index])
        x_pad = np.array([self.pad_face_token_in for _ in range(0, self.block_size - len(x_in))])
        y_pad = np.array([self.pad_face_token_tgt for _ in range(0, self.block_size - len(x_in))])
        x_in = np.hstack((x_in, x_pad))
        y_in = np.hstack((y_in, y_pad))
        x = torch.from_numpy(x_in.astype(np.int64))
        y = torch.from_numpy(y_in.astype(np.int64))

        return {
            'name': self.names[index],
            'input': x,
            'target': y,
        }

    def get_completion_sequence(self, i, tokens, device=torch.device("cpu")):
        vertices = normalize_vertices(self.cached_vertices[i])
        faces = self.cached_faces[i]
        soup_sequence = quantize_soup(vertices, faces, num_tokens=self.num_tokens - 3, max_vertices=self.max_vertices, max_faces=self.max_faces)  # throws error if processing fails

        original_fseq_text = " ".join(map(str, soup_sequence.tolist()))
        original_fseq = self.tokenizer.encode(original_fseq_text).tolist()
        original_fseq.append(self.tokenizer.eos_id)

        if isinstance(tokens, int):
            num_pre_tokens = tokens
        else:
            num_pre_tokens = int(len(original_fseq) * tokens)

        x = (
            torch.tensor(original_fseq[:num_pre_tokens], dtype=torch.int64, device=device)[None, ...],
            torch.tensor(original_fseq, dtype=torch.int64, device=device)[None, ...],
        )
        return x

    def get_start(self, device=torch.device("cpu")):
        i = random.choice(list(range(len(self.cached_vertices))))
        x = self.get_completion_sequence(i, 44, device)
        return x

    def __len__(self) -> int:
        return len(self.cached_vertices)

    def decode(self, sequence):
        return ngonsoup_sequence_to_mesh(sequence, self.tokenizer, self.vocab_size - 3)

    def plot_sequence_lenght_stats(self):
        sequence_lengths = []
        for i in range(len(self.cached_faces)):
            sequence_len = len([x for fl in self.cached_faces[i] for x in fl]) * 3 + len(self.cached_faces[i]) + 1
            sequence_lengths.append(sequence_len)
        import matplotlib.pyplot as plt
        plt.hist(sequence_lengths, bins=32)
        plt.ylim(0, 100)
        plt.show()


def test_soup_dataset():
    from easydict import EasyDict
    from tqdm import tqdm
    from lit_gpt import Tokenizer

    tokenizer = Tokenizer(Path("/cluster/gimli/ysiddiqui/Llama-2-7b-hf"))

    dataset = NgonSoup(
        tokenizer,
        EasyDict(  # type: ignore
            {
                'block_size': 4096,
                # 'dataset_root': 'data/shapenet/chairsoup_256_simple000.pkl',
                'dataset_root': 'data/shapenet/soup_256_1.5K_4K.pkl',
                # 'dataset_root': 'data/shapenet/ngonsoup_256_1.5K_4K.pkl',
                'overfit': True,
                'num_tokens': 259,
                'padding': 0,
                'scale_augment': True,
                'max_vertices': 2500,
                'max_faces': 10000
            }
        ), 'train', scale_augment=True)

    dataset.plot_sequence_lenght_stats()
    output_meshes = True

    if output_meshes:
        for i in tqdm(list(range(100)), 'testing speed'):
            access = dataset[i]['input']

        for i in tqdm(list(range(5))):
            sample = dataset[0]
            print(dataset[0]['name'], sample['input'].shape)
            gen_vertices, gen_faces = dataset.decode(sample['input'])
            plot_vertices_and_faces(gen_vertices, gen_faces, Path(f"runs/test_{i:02d}.jpg"))
            trimesh.Trimesh(vertices=gen_vertices, faces=gen_faces, process=False).export(f"runs/test_{i:02d}.obj")

    dataset = NgonSoup(
        tokenizer,
        EasyDict(  # type: ignore
            {
                'block_size': 8,
                'dataset_root': 'data/shapenet/soup_256_1.5K_4K.pkl',
                'overfit': False,
                'num_tokens': 259,
                'padding': 0,
                'scale_augment': True,
                'max_vertices': 2500,
                'max_faces': 10000
            }
        ), 'train', scale_augment=False)

    for i in [0, 1, 2, 3]:
        sample = dataset[i]
        print(sample['input'].shape, '\n')


if __name__ == "__main__":
    # test dataset
    test_soup_dataset()
