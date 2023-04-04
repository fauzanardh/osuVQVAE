import random
import numpy as np

import torch
from torch.utils.data import IterableDataset


def load_tensor(map_file):
    x = torch.tensor(np.load(map_file)["x"]).float()
    # a = torch.tensor(np.load(map_file.parent / "spec.npz")["spec"]).float()
    return x


class StreamPerSample(IterableDataset):
    def __init__(self, **kwargs):
        super().__init__()

        self.dataset = kwargs.pop("dataset")
        self.sample_density = kwargs.pop("sample_density", 1.0)

        if not (0 < self.sample_density <= 1.0):
            raise ValueError("sample_density must be in (0, 1]")

    def sample_stream(self, map_file):
        raise NotImplementedError

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_workers = 1
            worker_id = 0
            seed = torch.initial_seed()
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            seed = worker_info.seed

        random.seed(seed)

        dataset = sorted(self.dataset)
        for i, sample in random.sample(
            list(enumerate(dataset)), int(len(dataset) * self.sample_density)
        ):
            if i % num_workers != worker_id:
                continue

            for x in self.sample_stream(sample):
                yield x


class FullSequenceDataset(StreamPerSample):
    MAX_LEN = 60000

    def sample_stream(self, map_file):
        x = load_tensor(map_file)
        yield x[..., : self.MAX_LEN]


class SubsequenceDataset(StreamPerSample):
    def __init__(self, **kwargs):
        self.seq_len = kwargs.pop("seq_len")
        self.subseq_density = kwargs.pop("subseq_density", 2)
        super().__init__(**kwargs)

        # num_samples = 0
        # for map_file in self.dataset:
        #     with open(map_file, "rb") as f:
        #         magic = np.lib.format.read_magic(f)
        #         read_header = (
        #             np.lib.format.read_array_header_1_0
        #             if magic == 1
        #             else np.lib.format.read_array_header_2_0
        #         )
        #         shape = read_header(f)[0]
        #         num_samples += int(shape[-1] / self.seq_len * self.subseq_density)

        # self.approx_dataset_size = num_samples * self.subseq_density

    def sample_stream(self, map_file):
        x = load_tensor(map_file)
        n = x.shape[-1]

        if self.seq_len >= n:
            return

        num_samples = int(n / self.seq_len * self.subseq_density)
        for idx in torch.randperm(n - self.seq_len)[:num_samples]:
            yield x[..., idx : idx + self.seq_len].clone()
