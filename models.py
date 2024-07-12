
import json
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
import torch.nn.functional as F
import bitsandbytes as bnb

from torch import nn
from joblib import load
from peft import LoraModel, LoraConfig
from datagpt import Set, system_prompt, confidence_prompt, ROWS, MIN, MAX
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM

from scipy.signal import savgol_filter

ALL_ROWS = {
    "browInnerUpLeft":     35.426251,
    "browInnerUpRight":    34.682091,
    "browDownLeft":        11.193328,
    "browDownRight":        9.448617,
    "browOuterUpLeft":      5.489865,
    "browOuterUpRight":     5.624059,
    "eyeLookUpLeft":        6.735112,
    "eyeLookUpRight":       6.714617,
    "eyeLookDownLeft":      6.425929,
    "eyeLookDownRight":     6.118277,
    "eyeLookInLeft":       12.934639,
    "eyeLookInRight":      10.825693,
    "eyeLookOutLeft":       7.103837,
    "eyeLookOutRight":     14.446621,
    "eyeBlinkLeft":         5.644818,
    "eyeBlinkRight":        5.644818,
    "eyeSquintLeft":        0.258453,
    "eyeSquintRight":       0.205676,
    "eyeWideLeft":          9.493767,
    "eyeWideRight":         9.634814,
    "cheekPuffLeft":        2.920025,
    "cheekPuffRight":       2.536376,
    "cheekSquintLeft":      4.761187,
    "cheekSquintRight":     4.457829,
    "noseSneerLeft":        2.892463,
    "noseSneerRight":       3.933417,
    "jawOpen":              0.000000,
    "jawForward":           0.000000,
    "jawLeft":              0.000000,
    "jawRight":             0.000000,
    "mouthFunnel":          0.000000,
    "mouthPucker":          0.000000,
    "mouthLeft":           -1.525280,
    "mouthRight":          -0.588367,
    "mouthRollUpper":       0.000000,
    "mouthRollLower":       0.000000,
    "mouthShrugUpper":      1.164939,
    "mouthShrugLower":      1.164939,
    "mouthClose":           0.172632,
    "mouthSmileLeft":       3.817821,
    "mouthSmileRight":      3.172182,
    "mouthFrownLeft":       3.568846,
    "mouthFrownRight":      3.596487,
    "mouthDimpleLeft":      1.908910,
    "mouthDimpleRight":     1.586091,
    "mouthUpperUpLeft":     0.620497,
    "mouthUpperUpRight":    0.933096,
    "mouthLowerDownLeft":  -0.517081,
    "mouthLowerDownRight": -0.777580,
    "mouthPressLeft":       0.000000,
    "mouthPressRight":      0.000000,
    "mouthStretchLeft":     0.000000,
    "mouthStretchRight":    0.000000,
    "tongueOut":            0.000000,
    "headRotationX":        0.019367,
    "headRotationY":        0.024757,
    "headRotationZ":        0.017393,
    "headRotationW":        0.998417,
    "eyeLeftRotationX":     0.689709,
    "eyeLeftRotationY":    -0.018976,
    "eyeLeftRotationZ":    -0.043799,
    "eyeLeftRotationW":    -0.714934,
    "eyeRightRotationX":    0.689709,
    "eyeRightRotationY":   -0.018976,
    "eyeRightRotationZ":   -0.043799,
    "eyeRightRotationW":   -0.714934,
}


def mse_loss(x, y, w):
    assert x.shape == y.shape
    assert x.shape[:-1] == w.shape
    d = torch.square(x - y) * w.unsqueeze(-1)
    l = d.sum(dim=-1).sum(dim=-1) / w.sum(dim=-1)
    return l.mean()


class Conv1DBuilder(object):
    @staticmethod
    def build(in_channels, out_channels, kernel_size, stride=1, padding=0, use_kaiming_normal=False, bias=True):
        conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        if use_kaiming_normal:
            conv = nn.utils.weight_norm(conv)
            nn.init.kaiming_normal_(conv.weight)
        return conv


class ConvTranspose1DBuilder(object):
    @staticmethod
    def build(in_channels, out_channels, kernel_size, stride=1, padding=0, use_kaiming_normal=False, bias=True):
        conv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        if use_kaiming_normal:
            conv = nn.utils.weight_norm(conv)
            nn.init.kaiming_normal_(conv.weight)
        return conv


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens, use_kaiming_normal):
        super(Residual, self).__init__()
        
        relu_1 = nn.ReLU()
        conv_1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=num_residual_hiddens,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )
        # bn_1 = nn.BatchNorm1d(num_residual_hiddens)
        bn_1 = nn.Identity()
        if use_kaiming_normal:
            conv_1 = nn.utils.weight_norm(conv_1)
            nn.init.kaiming_normal_(conv_1.weight)

        relu_2 = nn.ReLU()
        conv_2 = nn.Conv1d(
            in_channels=num_residual_hiddens,
            out_channels=num_hiddens,
            kernel_size=1,
            stride=1,
            bias=True
        )
        # bn_2 = nn.BatchNorm1d(num_hiddens)
        bn_2 = nn.Identity()
        if use_kaiming_normal:
            conv_2 = nn.utils.weight_norm(conv_2)
            nn.init.kaiming_normal_(conv_2.weight)

        # All parameters same as specified in the paper
        self._block = nn.Sequential(
            relu_1,
            conv_1,
            bn_1,
            relu_2,
            conv_2,
            bn_2
        )
    
    def forward(self, x):
        return x + self._block(x)


class DownscaleBlock(nn.Module):
    def __init__(self, num_hiddens, use_kaiming_normal):
        super().__init__()
        
        self._conv_3 = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=4,
            stride=2, # timestep * 2
            use_kaiming_normal=use_kaiming_normal,
            padding=2,
            bias=True
        )
        # self.bn_3 = nn.BatchNorm1d(num_hiddens)
        self.bn_3 = nn.Identity()
        self._conv_4 = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal,
            padding=1,
            bias=True
        )
        # self.bn_4 = nn.BatchNorm1d(num_hiddens)
        self.bn_4 = nn.Identity()
        self._conv_5 = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal,
            padding=1,
            bias=True
        )
        # self.bn_5 = nn.BatchNorm1d(num_hiddens)
        self.bn_5 = nn.Identity()

    def forward(self, x):
        x_conv_3 = F.relu(self.bn_3(self._conv_3(x)))
        x_conv_4 = F.relu(self.bn_4(self._conv_4(x_conv_3))) + x_conv_3
        x_conv_5 = F.relu(self.bn_5(self._conv_5(x_conv_4))) + x_conv_4
        return x_conv_5


class UpscaleBlock(nn.Module):
    def __init__(self, num_hiddens, use_kaiming_normal):
        super().__init__()
        self._upsample = nn.Upsample(scale_factor=2)
        self._conv_trans_1 = ConvTranspose1DBuilder.build(
            in_channels=num_hiddens, 
            out_channels=num_hiddens,
            kernel_size=3,
            padding=1,
            use_kaiming_normal=use_kaiming_normal,
            bias=True
        )
        # self.bn_1 = nn.BatchNorm1d(num_hiddens)
        self.bn_1 = nn.Identity()
        self._conv_trans_2 = ConvTranspose1DBuilder.build(
            in_channels=num_hiddens, 
            out_channels=num_hiddens,
            kernel_size=3,
            padding=0,
            use_kaiming_normal=use_kaiming_normal,
            bias=True
        )
        # self.bn_2 = nn.BatchNorm1d(num_hiddens)
        self.bn_2 = nn.Identity()
        self._conv_trans_3 = ConvTranspose1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=2,
            padding=0,
            use_kaiming_normal=use_kaiming_normal,
            bias=True
        )
        # self.bn_3 = nn.BatchNorm1d(num_hiddens)
        self.bn_3 = nn.Identity()

    def forward(self, x):
        x = self._upsample(x)
        x = F.relu(self.bn_1(self._conv_trans_1(x)))
        x = F.relu(self.bn_2(self._conv_trans_2(x)))
        x = F.relu(self.bn_3(self._conv_trans_3(x)))
        return x


class ConvolutionalEncoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, latent_dim, num_residual_layers, num_residual_hiddens, use_kaiming_normal):
        super(ConvolutionalEncoder, self).__init__()
        self._conv_1 = Conv1DBuilder.build(
            in_channels=in_channels,
            out_channels=num_hiddens,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal,
            padding=1
        )
        self._conv_2 = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal,
            padding=1
        )

        self.blocks = nn.Sequential(
            DownscaleBlock(num_hiddens, use_kaiming_normal),
            DownscaleBlock(num_hiddens, use_kaiming_normal),
            DownscaleBlock(num_hiddens, use_kaiming_normal),
            DownscaleBlock(num_hiddens, use_kaiming_normal),
        )
        
        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            use_kaiming_normal=use_kaiming_normal
        )

    def forward(self, inputs):
        x_conv_1 = F.relu(self._conv_1(inputs))
        x = F.relu(self._conv_2(x_conv_1)) + x_conv_1
        
        x = self.blocks(x)
        x = self._residual_stack(x) + x
        
        o = torch.flatten(x, 1)
        return o, x.shape


class DeconvolutionalDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_hiddens, num_residual_layers,
                 num_residual_hiddens, use_kaiming_normal):
        super(DeconvolutionalDecoder, self).__init__()

        self._conv_1 = Conv1DBuilder.build(
            in_channels=in_channels,
            out_channels=num_hiddens,
            kernel_size=3,
            padding=1,
            use_kaiming_normal=use_kaiming_normal
        )
        self.upscale = nn.Sequential(
            UpscaleBlock(num_hiddens, use_kaiming_normal),
            UpscaleBlock(num_hiddens, use_kaiming_normal),
            UpscaleBlock(num_hiddens, use_kaiming_normal),
            UpscaleBlock(num_hiddens, use_kaiming_normal),
        )
        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            use_kaiming_normal=use_kaiming_normal
        )
        self.seq_output = Conv1DBuilder.build(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            use_kaiming_normal=use_kaiming_normal
        )

    def forward(self, x, shape):
        x = torch.reshape(x, (x.shape[0], shape[1], shape[2]))
        
        x = self._conv_1(x)
        x = self.upscale(x)
        x = self._residual_stack(x)
        seq = self.seq_output(x)
        return seq


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, use_kaiming_normal):
        super(ResidualStack, self).__init__()
        
        self._num_residual_layers = num_residual_layers
        self._layers = nn.Sequential(
            *[Residual(in_channels, num_hiddens, num_residual_hiddens, use_kaiming_normal) for _ in range(self._num_residual_layers)]
        )
        
    def forward(self, x):
        return torch.nn.functional.relu(self._layers(x))


class SequenceEncoder(nn.Module):
    def __init__(self, feature_dim, hidden_dim, latent_dim, residual_layers):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        self.encoder = ConvolutionalEncoder(
            in_channels=feature_dim,
            num_hiddens=hidden_dim,
            latent_dim=latent_dim,
            num_residual_layers=residual_layers,
            num_residual_hiddens=hidden_dim * 1,
            use_kaiming_normal=False
        )
        
    def forward(self, x):
        x = x.transpose(-1, -2)
        ht = self.encoder(x)
        return ht


class SequenceDecoder(nn.Module):
    def __init__(self, feature_dim, hidden_dim, latent_dim, residual_layers):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        self.decoder = DeconvolutionalDecoder(
            in_channels=latent_dim,
            out_channels=hidden_dim,
            num_hiddens=hidden_dim,
            num_residual_layers=residual_layers,
            num_residual_hiddens=hidden_dim * 1,
            use_kaiming_normal=False,
        )
        self.output = nn.Linear(latent_dim, feature_dim)
    
    def forward(self, input_sequence, x, x_mask, shape):
        output_sequence = self.generate(x, shape)
        output_sequence = output_sequence[..., :1800]
        
        seq_loss = mse_loss(input_sequence, output_sequence.transpose(-1, -2), x_mask)
        return output_sequence, seq_loss

    def generate(self, sequence, shape):
        sequence = self.decoder(sequence, shape)
        sequence = self.output(sequence.transpose(-1, -2))
        return sequence.transpose(-1, -2)


class Sample(nn.Module):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, mean, log_var):
        return mean + torch.exp(0.5 * log_var) * torch.randn_like(log_var)
    

class VAE(nn.Module):
    def __init__(self, feature_dim, latent_dim, hidden_dim, residual_layers, inner_dimension=114):
        super().__init__()
        self.beta = 0.5
        self.encoder = SequenceEncoder(feature_dim, hidden_dim, hidden_dim, residual_layers)
        self.decoder = SequenceDecoder(feature_dim, hidden_dim, hidden_dim, residual_layers)
        self.sequence_decoder = nn.Sequential(
            nn.Linear(hidden_dim * inner_dimension + latent_dim, hidden_dim * inner_dimension),
            # nn.ReLU()
        )
        self.sample = Sample()
        self.mu = nn.Linear(
            hidden_dim * inner_dimension, hidden_dim * inner_dimension
        )
        self.sigma = nn.Linear(
            hidden_dim * inner_dimension, hidden_dim * inner_dimension
        )
        
    def generate(self, z, meta_latent, shape):
        z = torch.cat([z, meta_latent], dim=-1)
        sequence = self.sequence_decoder(z)
        sequence = self.decoder.generate(sequence, shape)
        return sequence[..., :1800].transpose(-1, -2)
        
    def forward(self, meta_latent, input_sequence, input_masks):
        ht, shape = self.encoder(input_sequence)

        mean, log_var = self.mu(ht), self.sigma(ht)
        z = self.sample(mean, log_var)

        # Apply confidence
        # z, latent_loss = self.sample_with_confidence(z, confidence)
        z = torch.cat([z, meta_latent], dim=-1)

        sequence = self.sequence_decoder(z)
        sequence, rec_loss = self.decoder(input_sequence, sequence, input_masks, shape)
        
        kl_loss = -0.5 * (1 + log_var - torch.square(mean) - torch.exp(log_var)).sum(dim=-1) * self.beta
        return sequence, kl_loss.mean(), rec_loss.mean() * 10


class LengthPredictor(pl.LightningModule):
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.pre_pause_predictor = nn.Sequential(
            nn.Linear(12, 32),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.ReLU()
        )
        self.post_pause_predictor = nn.Sequential(
            nn.Linear(12, 32),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.ReLU()
        )

    def predict(self, input):
        with torch.no_grad():
            pre_prediction = self.pre_pause_predictor(input)
            post_prediction = self.pre_pause_predictor(input)
            pre_prediction = int(np.exp(pre_prediction.cpu().numpy()) * 60)
            post_prediction = int(np.exp(post_prediction.cpu().numpy()) * 60)
            return pre_prediction, post_prediction

    def training_step(self, batch, batch_idx):
        input, pre_length, post_length = batch['encoding'], batch['log_pre_length'], batch['log_post_length']
        # ground_truth = torch.stack([pre_length, post_length], dim=-1)
        pre_prediction = self.pre_pause_predictor(input)
        post_prediction = self.pre_pause_predictor(input)

        loss = F.mse_loss(pre_prediction, pre_length[..., None]) + F.mse_loss(post_prediction, post_length[..., None])
        self.log("loss", loss.item(), prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input, pre_length, post_length = batch['encoding'], batch['log_pre_length'], batch['log_post_length']
        # ground_truth = torch.stack([pre_length, post_length], dim=-1)
        pre_prediction = self.pre_pause_predictor(input)
        post_prediction = self.pre_pause_predictor(input)

        loss = F.mse_loss(pre_prediction, pre_length[..., None]) + F.mse_loss(post_prediction, post_length[..., None])
        self.log("val_loss", loss.item(), prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [{
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }, {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        # return bnb.optim.AdamW8bit(optimizer_grouped_parameters, lr=self.learning_rate)
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class Baseline(pl.LightningModule):
    def __init__(self, learning_rate: float = 5e-4, model: str="", 
                 lora_config: LoraConfig|None = None, quantize: bool = False,
                 sys_prompt:str = system_prompt()):
        super().__init__()
        self.save_hyperparameters()
        self.lora_config = lora_config
        self.learning_rate = learning_rate
        self.quantize = quantize
        self.system_prompt = sys_prompt
        self.model_name = model

        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained(model, torch_dtype=torch.float16, token="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") # , bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
        self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
    
    def configure_model(self):
        if self.model is not None:
            return
        
        if self.lora_config is not None:
            if self.quantize:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    token="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",
                    load_in_4bit=True,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    max_memory= {i: '48000MB' for i in range(torch.cuda.device_count())},
                    quantization_config=BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type='nf4'
                    ),
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, device_map="cpu", token="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")
            self.model = LoraModel(model, self.lora_config, "default")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, token="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")

    def prompt(self, confidence) -> str:
        prompt = f"""
<|system|>
{system_prompt()}</s>
<|user|>
{confidence_prompt(confidence * 100)}</s>
<|assistant|>"""
        return prompt
    
    def training_step(self, batch, batch_idx):
        # Note that data_input and data_masks encodes the tokens including the bytes of the floats
        idx, mask = batch['data_input'], batch['data_masks']
        output = self.model(idx, attention_mask=mask, labels=idx)

        self.log("loss", output.loss.item(), prog_bar=True, on_epoch=True)
        return output.loss

    def validation_step(self, batch, batch_idx):
        # Note that data_input and data_masks encodes the tokens including the bytes of the floats
        idx, mask = batch['data_input'], batch['data_masks']
        output = self.model(idx, attention_mask=mask, labels=idx)

        self.log("val_loss", output.loss.item(), prog_bar=True, on_epoch=True)
        return output.loss
        
    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [{
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.001,
            }, {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return bnb.optim.AdamW8bit(optimizer_grouped_parameters, lr=self.learning_rate)


class BlendShapeVAE(pl.LightningModule):
    num_files = 0

    def __init__(self, feature_dim, hidden_dim, latent_dim, residual_layers, inner_dimension,
                 learning_rate=5e-4, conf_method="none",
                 model: str="", model_dim: int=0,
                 lora_config: LoraConfig|None = None, quantize: bool = False,
                 sys_prompt:str = system_prompt()):
        super().__init__()
        self.save_hyperparameters()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.confidence_method = conf_method
        self.residual_layers = residual_layers
        self.quantize = quantize
        self.system_prompt = sys_prompt
        self.inner_dimension = inner_dimension
        self.cache = {}

        self.tokenizer = AutoTokenizer.from_pretrained(model, token="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") # , bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
        self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
        
        self.pre_regressor = load("pre_regressor.jlib")
        self.post_regressor = load("post_regressor.jlib")

        if lora_config is not None:
            if self.quantize:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model,
                    token="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",
                    load_in_4bit=True,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    max_memory= {i: '48000MB' for i in range(torch.cuda.device_count())},
                    quantization_config=BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type='nf4'
                    ),
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model, token="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")
            self.model = LoraModel(self.model, lora_config, "default")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model, token="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")

        self.vae = VAE(feature_dim, model_dim, hidden_dim, residual_layers, inner_dimension)

    def prompt(self, confidence) -> str:
        prompt = f"""
<|system|>
{system_prompt()}</s>
<|user|>
{confidence_prompt(confidence * 100)}</s>
<|assistant|>"""
        return prompt

    def generate(self, confidence, mean, std, use_regressor) -> tuple[pd.DataFrame, str, str, str, str]:
        with torch.no_grad():
            prompt = self.prompt(confidence)
            generated = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0)
            generated = generated.to(self.model.device)
            sample_outputs = self.model.generate(
                generated,
                temperature=1.1,
                top_p=0.98,
                do_sample=True,
            )
            decoded = self.tokenizer.decode(sample_outputs[0], skip_special_tokens=True)

            out = decoded.strip().replace("</s>", "")
            generated = json.loads(out.split("<|assistant|>")[1])
            with open(f"dummy/meta-llama-{int(confidence*100)}-{BlendShapeVAE.num_files}.json", "w") as file:
                json.dump(generated, file)

            if use_regressor:
                metadata, generated = self.convert(confidence, decoded)
                if metadata is None:
                    return None, None
                
                _, intonation, filler, pre_hedge, post_hedge, pre_length, perform_length, post_length = metadata
                input = [
                    intonation == "falling", intonation  == "rising",
                    filler == "none", filler == "uh", filler == "um",
                    pre_hedge == "ithink", pre_hedge == "maybe", pre_hedge == "none",
                    post_hedge == "butimnotsure", post_hedge == "idontknow", post_hedge == "none"
                ]
                if np.sum(input) != 4:
                    return None, None
                
                pre_length = np.exp(self.pre_regressor.predict([[int(confidence*100)] + input])) * 60
                post_length = np.exp(self.post_regressor.predict([[int(confidence*100)] + input])) * 60

                decoded = Set.prompt_from_metadata(
                    confidence*100, intonation, filler, pre_hedge, post_hedge, int(pre_length), int(perform_length), int(post_length)
                )

                out = decoded.strip().replace("</s>", "")
                generated = json.loads(out.split("<|assistant|>")[1])
                global num_files
                with open(f"dummy/meta-regressor-{int(confidence*100)}-{BlendShapeVAE.num_files}.json", "w") as file:
                    json.dump(generated, file)


            BlendShapeVAE.num_files += 1
            tokens = self.tokenizer(decoded, truncation=False, max_length=512, padding='max_length')
            
            token_ids = torch.tensor(tokens['input_ids'], device=self.model.device)[None, ...]
            token_masks = torch.tensor(tokens['attention_mask'], device=self.model.device)[None, ...]
            
            outputs = self.model(token_ids, attention_mask=token_masks, output_hidden_states=True)
            latent = outputs.hidden_states[-1]
            lengths = torch.sum(token_masks, dim=-1)
            
            meta_latents = torch.sum(latent * token_masks[..., None], dim=1) / lengths[..., None]
            epsilon = torch.normal(mean=torch.zeros((1, self.hidden_dim * self.inner_dimension), device=self.model.device), std=torch.ones((1, self.hidden_dim * self.inner_dimension), device=self.model.device))
            z = mean + np.exp(0.5 * std) * epsilon

            sequence = self.vae.generate(z, meta_latents, shape=(1, self.hidden_dim, self.inner_dimension)) 
            sequence = sequence.cpu().numpy() * (MAX - MIN) + MIN
            # for i, seq in enumerate(sequence):
            
            sequences = [pd.DataFrame(x, columns=ROWS) for x in sequence]
            for i, sequence in enumerate(sequences):
                for name, val in ALL_ROWS.items():
                    if name not in sequence.columns:
                        sequence[name] = val
                sequences[i] = sequence
            
            sequence["browInnerUpLeft"]  = savgol_filter(sequence["browInnerUpLeft"], 31, 3)
            sequence["browInnerUpRight"] = savgol_filter(sequence["browInnerUpRight"], 31, 3)
            sequence["browDownLeft"]     = savgol_filter(sequence["browDownLeft"], 31, 3)
            sequence["browDownRight"]    = savgol_filter(sequence["browDownRight"], 31, 3)
            sequence["browOuterUpLeft"]  = savgol_filter(sequence["browOuterUpLeft"], 31, 3)
            sequence["browOuterUpRight"] = savgol_filter(sequence["browOuterUpRight"], 31, 3)

            sequence["headRotationX"] = savgol_filter(sequence["headRotationX"], 101, 3)
            sequence["headRotationY"] = savgol_filter(sequence["headRotationY"], 101, 3)
            sequence["headRotationZ"] = savgol_filter(sequence["headRotationZ"], 101, 3)
            sequence["headRotationW"] = savgol_filter(sequence["headRotationW"], 101, 3)
            return sequence, decoded
        
    def convert(self, confidence, decoded):
        try:
            generated = json.loads(decoded.split("<|assistant|>")[1])
            confidence, intonation, filler, pre_hedge, post_hedge, pre_length, perform_length, post_length = \
                confidence, generated["intonation"], generated["filler"], \
                generated["pre_hedge"], generated["post_hedge"], generated["pre_length"], generated["perform_length"], generated["post_length"]
            return (confidence, intonation, filler, pre_hedge, post_hedge, pre_length, perform_length, post_length), generated
        except Exception as e:
            return None, None

    def training_step(self, batch, batch_idx):
        idx, mask = batch['meta_input_ids'], batch['meta_attn_masks']
        
        gpt_loss = 0
        if self.current_epoch > 5:
            latents = []
            for i, batch_index in enumerate(batch["index"]):
                if batch_index.item() not in self.cache:
                    with torch.no_grad():
                        outputs = self.model(idx[i:i+1], attention_mask=mask[i:i+1], labels=idx[i:i+1], output_hidden_states=True)
                        latent = outputs.hidden_states[-1]
                        lengths = torch.sum(mask, dim=-1)
                        meta_latents = torch.sum(latent * mask[..., None], dim=1) / lengths[..., None]
                        self.cache[batch_index.item()] = meta_latents.clone().detach()[0]
                else:
                    assert self.current_epoch != 0, "This can not be since we are in the initial phase still there should not be an index!"

                latents.append(self.cache[batch_index.item()])
            meta_latents = torch.stack(latents, dim=0)
        else:
            outputs = self.model(idx, attention_mask=mask, labels=idx, output_hidden_states=True)
            gpt_loss = outputs.loss
            self.log("gpt", gpt_loss.item(), prog_bar=True, on_epoch=True)

            latent = outputs.hidden_states[-1]
            lengths = torch.sum(mask, dim=-1)
            meta_latents = torch.sum(latent * mask[..., None], dim=1) / lengths[..., None]
            meta_latents = meta_latents.detach()
            
        x, mask = batch['data_input'], batch['data_masks']
        _, kl_loss, rec_loss = self.vae.forward(meta_latents, x, mask)

        loss = rec_loss + kl_loss + gpt_loss
        self.log("rec", rec_loss.item(), prog_bar=True, on_epoch=True)
        self.log("loss", loss.item(), prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        idx, mask = batch['meta_input_ids'], batch['meta_attn_masks']
        
        # with torch.no_grad():
        outputs = self.model(idx, attention_mask=mask, labels=idx, output_hidden_states=True)
        
        latent = outputs.hidden_states[-1]
        lengths = torch.sum(mask, dim=-1)
        meta_latents = torch.sum(latent * mask[..., None], dim=1) / lengths[..., None]
        
        x, mask = batch['data_input'], batch['data_masks']
        _, kl_loss, rec_loss = self.vae.forward(meta_latents.detach(), x, mask)

        loss = rec_loss + kl_loss + outputs.loss
        self.log("val_gpt", outputs.loss.item(), prog_bar=True, on_epoch=True)
        self.log("val_rec", rec_loss.item(), prog_bar=True, on_epoch=True)
        self.log("val_loss", loss.item(), prog_bar=True, on_epoch=True)
        return loss
        
    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [{
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.001,
            }, {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return bnb.optim.AdamW8bit(optimizer_grouped_parameters, lr=self.learning_rate)