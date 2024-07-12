from __future__ import annotations

import os, json
import torch
import numpy as np
import pandas as pd
import lightning.pytorch as pl

from transformers import GPT2Tokenizer, AutoTokenizer
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Optional
from tqdm import tqdm

ROWS = ["browInnerUpLeft", "browInnerUpRight", "browDownLeft", "browDownRight", "browOuterUpLeft", "browOuterUpRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "cheekPuffLeft", "cheekPuffRight", "cheekSquintLeft", "cheekSquintRight", "noseSneerLeft", "noseSneerRight", "mouthLeft", "mouthRight", "mouthShrugUpper", "mouthShrugLower", "mouthClose", "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight", "mouthDimpleLeft", "mouthDimpleRight", "mouthUpperUpLeft", "mouthUpperUpRight", "mouthLowerDownLeft", "mouthLowerDownRight", "headRotationX", "headRotationY", "headRotationZ", "headRotationW", "eyeLeftRotationX", "eyeLeftRotationY", "eyeLeftRotationZ", "eyeLeftRotationW", "eyeRightRotationX", "eyeRightRotationY", "eyeRightRotationZ", "eyeRightRotationW"]
MEAN = np.array([35.42625,34.68209,11.193328,9.448617,5.489865,5.624059,6.7351117,6.7146173,6.4259295,6.1182766,12.934639,10.825693,7.103837,14.446621,5.6448183,5.6448183,0.2584528,0.20567615,9.493767,9.634814,2.9200249,2.5363758,4.7611876,4.4578295,2.892463,3.9334168,-1.52528,-0.5883671,1.1649386,1.1649386,0.17263222,3.817821,3.1721816,3.5688465,3.596487,1.9089105,1.5860908,0.62049735,0.933096,-0.51708114,-0.77758,0.019366555,0.024757005,0.017393207,0.998417,0.6897093,-0.018975902,-0.04379949,-0.71493405,0.6897093,-0.018975902,-0.04379949,-0.71493405], dtype=np.float32)
VAR  = np.array([52.45966,52.824757,22.530165,20.881401,13.870541,13.955491,14.511868,14.481247,15.98498,15.717793,24.745518,18.958235,14.621764,21.894785,18.127039,18.127039,8.58098,8.331993,21.394463,22.009218,9.6450815,8.8275,11.704032,10.513238,6.506281,9.2519655,3.0471015,1.1358262,2.6952958,2.6952958,0.3419568,8.694366,8.3888235,12.9827,9.892623,4.347183,4.1944118,1.700042,2.4352992,1.4167017,2.029416,0.033240832,0.023862833,0.013948744,0.0020883488,0.03044789,0.06612098,0.06706293,0.03310573,0.03044789,0.06612098,0.06706293,0.03310573], dtype=np.float32)

MIN = np.array([-43.04679,-43.42719,-29.22112,-40.90203,-15.06938,-15.1839,-0.6328122,-0.6262041,-30.08919,-29.76008,-5.946578,-3.707915,-0.2945891,-1.122671,-24.77933,-24.77933,-17.08037,-17.07599,-2.884468,-1.791656,-0.2679306,-0.6307207,-0.2617344,-0.1632695,-0.7676599,-0.9987831,-26.60651,-9.093588,-0.007197175,-0.007197175,-0.01289603,-0.1235052,-0.06439058,-19.71915,-20.08939,-0.06175258,-0.03219529,-0.2366741,-0.2996348,-22.10656,-24.38647,-0.11895,-0.04517,-0.05895,0.97909,0.57269,-0.30394,-0.30607,-0.80929,0.57269,-0.30394,-0.30607,-0.80929], dtype=np.float32)
MAX = np.array([202.0469,201.0377,132.4973,123.7119,70.36002,70.26895,98.53432,98.54633,98.83868,90.65083,99.50579,101.6544,89.5331,89.00202,102.8879,102.8879,68.81055,69.37666,98.33184,97.93076,89.15349,88.97007,89.94244,89.49109,106.6434,122.8172,0.03756836,0.04295971,19.60811,19.60811,2.811168,59.15743,60.26817,100.5541,77.59547,29.57872,30.13408,26.52787,29.26376,0.1972284,0.2496958,0.14746,0.19793,0.09723,0.99999,0.80094,0.1954,0.14343,-0.57217,0.80094,0.1954,0.14343,-0.57217], dtype=np.float32)

MAX_LENGTH = 0
MIN_LENGTH = 100000

MIN_META = 1000000
MAX_META = 0


def pad_sequence(sequence: torch.Tensor | np.ndarray, length: int, value=0.0) -> np.ndarray | torch.Tensor:
    seq_length = len(sequence)
    if len(sequence) == length:
        return sequence
    if len(sequence) > length:
        print(f"Length is {len(sequence)} but should be max {length}")
        assert False, "We don't have that here!"
        if truncating == PadSide.Left:
            return sequence[seq_length-length:]
        return sequence[:length]
    zeros = np.zeros((length-seq_length,) + sequence.shape[1:], dtype=sequence.dtype) + np.array([value], dtype=sequence.dtype)
    return np.concatenate([sequence, zeros], axis=0)


def fetch_metadata(directory: str) -> list[str]:
    return [os.path.join(directory, path) for path in os.listdir(directory) if path.lower().endswith(".json")] # [:10]


def parse_metadata(files: list[str]) -> list[dict[str, str]]:
    return [json.load(open(file, "r")) for file in files]


def parse_filenames_from_metadata(meta: list[dict[str, str]]) -> list[str]:
    return [data['sequence'] for data in meta]


def load_csv(directory, filename, FAKE_DATA:bool=False) -> pd.DataFrame:
    data = pd.read_csv(os.path.join(directory, filename), sep=",", header=0, index_col=False)[ROWS]
    return data


def load(directory: str, filename: str, metadata: dict[str, str], tokenizer: GPT2Tokenizer, mapping, return_bytes: bool) -> Set:
    if os.path.exists(os.path.join(directory, "NPY")):
        data = np.load(os.path.join(directory, "NPY", f"{filename}.npz"))
        pre, perform, post = data['pre'], data['perform'], data['post']
    else:
        pre, perform, post = load_csv(os.path.join(directory, "TXT"), f"{filename}_1.txt"), \
                             load_csv(os.path.join(directory, "TXT"), f"{filename}_2.txt"), \
                             load_csv(os.path.join(directory, "TXT"), f"{filename}_3.txt")
        pre, perform, post = pre.to_numpy(np.float32), perform.to_numpy(np.float32), post.to_numpy(np.float32)
                            
    set = Set(pre, perform, post, metadata, tokenizer, mapping, return_bytes)
    set.normalize("minmax")
    set.merge()
    sets = set.finalize()
    return sets


def system_prompt():
    return f"""You are a helpful metadata generator that generates data for visual and non visual uncertainty cues for virtual agents given a confidence value.
Here, an uncertainty value of 0 means that the virtual agent is really uncertain in its answer and does not know the answer.
A value of 100 means that the agent is really certain and does know the answer."""


def confidence_prompt(confidence):
    return f"Please generate metadata for the following confidence: {int(confidence)}."


class Set:
    PAD_VALUE: int = -1
    INTONATIONS: list[str] = ["rising", "falling"]
    FILLERS: list[str] = ["none", "um", "uh", "I think", "maybe", "but I am not sure", "I don't know"]
    MAX_LENGTH = 1800
    RETURN_FULL = False

    INTONATIONS: list[str] = ["rising", "falling"]
    FILLERS: list[str] = ["none", "um", "uh"]
    PRE_HEDGE: list[str] = ["none", "ithink", "maybe"]
    POST_HEDGE: list[str] = ["none", "butimnotsure", "idontknow"]
    
    def __init__(self, pre: np.ndarray, perform: np.ndarray, post: np.ndarray, meta: dict[str, str], tokenizer: AutoTokenizer, mapping: lambda x: x, return_bytes: bool = False) -> None:
        self._pre = pre
        self._perform = perform
        self._post = post
        self.mapping = mapping
        self.annotator = meta["annotator"]
        self.tokenizer = tokenizer
        self.return_bytes = return_bytes

        # TODO think about if return_bytes influences the length
        if pre is not None:
            self.lengths = [pre.shape[0], perform.shape[0], post.shape[0]]
            self.full_length = self.lengths[0] + self.lengths[1] + self.lengths[2] 
            assert self.full_length < self.MAX_LENGTH

            global MAX_LENGTH
            global MIN_LENGTH
            MAX_LENGTH = max(MAX_LENGTH, self.full_length)
            MIN_LENGTH = min(MIN_LENGTH, self.full_length)
        
            self._pre_mask = np.ones((pre.shape[0],), dtype=np.float32)
            self._perform_mask = np.ones((perform.shape[0],), dtype=np.float32)
            self._post_mask = np.ones((post.shape[0],), dtype=np.float32)

        self._data = None
        self._meta = meta
        
        self.intonation = self._meta['intonation']
        self.filler = self._meta['fillerWord']
        self.pre_hedge = self._meta['preHedge']
        self.post_hedge = self._meta['postHedge']
        self.pre_pause = self._meta['prePause']
        self.post_pause = self._meta['postPause']

        # if 'confidence' in self._meta:
        #     global MIN_META
        #     global MAX_META
        #     MIN_META = min(MIN_META, self._meta['confidence'])
        #     MAX_META = max(MAX_META, self._meta['confidence'])
        
    def normalize(self, method="std") -> None:
        match method:
            case "std": # Normalize to mean = 0 and var = 1
                self._pre = (self._pre - MEAN) / VAR
                self._perform = (self._perform - MEAN) / VAR
                self._post = (self._post - MEAN) / VAR
            case "minmax": # Normalize to 0-1
                self._pre = (self._pre - MIN) / (MAX - MIN)
                self._perform = (self._perform - MIN) / (MAX - MIN)
                self._post = (self._post - MIN) / (MAX - MIN)
            case _:
                assert False, "Unkown normlization method"

    def merge(self) -> None:
        self._data = np.concatenate([self._pre, self._perform, self._post], axis=0)
        self._pre_mask, self._perform_mask, self._post_mask = None, None, None
        self._pre, self._perform, self._post = None, None, None

    def finalize(self) -> list[Set]:
        if self.return_bytes:
            tokens = self.tokenizer(self.prompt(), truncation=False, padding='do_not_pad')

            output = []
            FRAMES = 6
            MAX_LENGTH = self._data.shape[-1] * 4 * FRAMES + 200 # ~ 6 frames which is the maximum we can support with our hardware
            for i in range(0, len(self._data), FRAMES):
                data = self._data[i:i+FRAMES]
                data = np.ascontiguousarray(data, data.dtype)
                sequence = data.view(np.uint8).reshape(-1,)

                data = np.concatenate([tokens["input_ids"], sequence])
                data_mask = np.ones((data.shape[0],), dtype=np.float32)

                data = pad_sequence(data, MAX_LENGTH, value=0)
                data_mask = pad_sequence(data_mask, MAX_LENGTH, value=0)

                set  = Set(self._pre, self._perform, self._post, self._meta, self.tokenizer, self.mapping, self.return_bytes)
                set._data = data
                set._data_mask = data_mask
                set.lengths = self.lengths
                output.append(set)
            return output
        else:
            self._data_mask = np.ones((self._data.shape[0],), dtype=np.float32)
            self._data_mask = pad_sequence(self._data_mask, self.MAX_LENGTH, value=0)
            self._data = pad_sequence(self._data, self.MAX_LENGTH, value=self.PAD_VALUE)
            return [self]

        # Check for invalid data 
        assert not np.any(np.isnan(self._data_mask))
        assert not np.any(np.isnan(self._data))

    @property
    def confidence(self) -> np.float32:
        return self._meta['confidence']

    @staticmethod
    def prompt_from_metadata(confidence, intonation, filler, pre_hedge, post_hedge, pre_length, perform_length, post_length):
        prompt = f"""
<|system|>
{system_prompt()}</s>
<|user|>
{confidence_prompt(confidence)}</s>
<|assistant|>
{{
    "intonation": "{intonation}",
    "filler": "{filler}",
    "pre_hedge": "{pre_hedge}",
    "post_hedge": "{post_hedge}",
    "pre_length": {pre_length},
    "perform_length": {perform_length},
    "post_length": {post_length}
}}</s>
"""
        return prompt

    def prompt(self) -> str:
        return self.prompt_from_metadata(self.confidence, self.intonation, self.filler, self.pre_hedge, self.post_hedge, self.lengths[0], self.lengths[1], self.lengths[2])

    @staticmethod
    def one_hot_data(confidence, intonation, filler, pre_hedge, post_hedge):
        confidence = np.zeros((1,))
        confidence[:] = confidence

        intonations = np.zeros((len(Set.INTONATIONS),))
        intonations[Set.INTONATIONS.index(intonation)] = 1

        fillers = np.zeros((len(Set.FILLERS),))
        fillers[Set.FILLERS.index(filler)] = 1

        pre_hedges = np.zeros((len(Set.PRE_HEDGE),))
        pre_hedges[Set.PRE_HEDGE.index(pre_hedge)] = 1

        post_hedges = np.zeros((len(Set.POST_HEDGE),))
        post_hedges[Set.POST_HEDGE.index(post_hedge)] = 1

        return np.concatenate([confidence, intonations, fillers, pre_hedges, post_hedges], axis=0).astype(np.float32)

    @property
    def data(self):
        tokens = self.tokenizer(self.prompt(), truncation=False, max_length=512, padding='max_length')
        return {
            'encoding': self.one_hot_data(self.confidence, self.intonation, self.filler, self.pre_hedge, self.post_hedge),
            'log_pre_length': np.log(self.pre_pause).astype(np.float32),
            'log_post_length': np.log(self.post_pause).astype(np.float32),
            'data_input': self._data,
            'data_masks': self._data_mask,
            'meta_input_ids': torch.tensor(tokens['input_ids']),
            'meta_attn_masks': torch.tensor(tokens['attention_mask']),
        }

    
class CustomDataset(Dataset):
    def __init__(self, directory: str, metadata: list[dict[str, str]], filenames: list[str], tokenizer: GPT2Tokenizer, mapping, return_bytes):
        self.sets = [load(directory, filename, meta, tokenizer, mapping, return_bytes) for filename, meta in zip(tqdm(filenames), metadata,)]
        self.sets = [set for superset in self.sets for set in superset]
        
    def __getitem__(self, index):
        data = self.sets[index].data
        data["index"] = index
        return data

    def __len__(self) -> int:
        return len(self.sets)


class BaselineDataset(Dataset):
    def __init__(self, directory: str, metadata: list[dict[str, str]], filenames: list[str], tokenizer: GPT2Tokenizer, mapping):
        self.sets = [load(directory, filename, meta, tokenizer, mapping) for filename, meta in zip(tqdm(filenames), metadata)]
        
    def __getitem__(self, index):
        data = self.sets[index].data
        data["index"] = index
        return data

    def __len__(self) -> int:
        return len(self.sets)


class BlendShapeDataset(pl.LightningDataModule):
    CPU_COUNT = min(multiprocessing.cpu_count(), 4) if os.name != "nt" else 0
    
    def __init__(self, directory: str, subdir: str, length: int, batch_size: int, synthetic:bool, tokenizer: str, return_bytes: bool = False):
        """
        return_bytes: If set to true the dataset will return byte tokens for the baseline
        """
        super().__init__()
        self.return_bytes = return_bytes
        self.mapping = lambda x: x
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, token="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") #, bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
        self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.synthetic = synthetic
        self.prepare_data_per_node = False # Otherwise we get in trouble because we don't have the data ready for other nodes

        self.directory = directory
        self.batch_size = batch_size
        self.length = length
        
        # def prepare_data(self) -> None:
        jsons = fetch_metadata(os.path.join(self.directory, subdir))
        self.meta  = parse_metadata(jsons)
        self.files = parse_filenames_from_metadata(self.meta)
        
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.full =  CustomDataset(
                directory=self.directory,
                metadata=self.meta,
                filenames=self.files,
                tokenizer=self.tokenizer,
                mapping=self.mapping,
                return_bytes=self.return_bytes
            )
            # Important, we do not 
            self.train, self.validate = random_split(self.full, [0.9, 0.1], generator=torch.Generator().manual_seed(42))
            
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=BlendShapeDataset.CPU_COUNT, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch_size, num_workers=BlendShapeDataset.CPU_COUNT)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=BlendShapeDataset.CPU_COUNT)

    

