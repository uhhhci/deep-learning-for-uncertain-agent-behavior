import argparse
import lightning.pytorch as pl

from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import TensorBoardLogger
from models import BlendShapeVAE, Baseline
from datagpt import BlendShapeDataset, ROWS
from lightning.pytorch.strategies import FSDPStrategy
from peft import LoraConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

import torch
torch.set_float32_matmul_precision('medium')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Training script to generate uncertain virtual agent behavior"
    )

    parser.add_argument("--batch_size", "--bs", type=int, default=1,
                        help="Batch size used to train the model")
    parser.add_argument("--accumulation_steps", "--acc_steps", type=int, default=20,
                        help="Accumulation steps over multiple batches before the optimization step")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help="Name of the base model for finetuning")
    parser.add_argument("--baseline", action="store_true", default=False,
                        help="If defined the baseline is trained")
    parser.add_argument("--learning_rate", "--lr", type=float, default=3e-4,
                        help="Learning rate for training the model")

    args = parser.parse_args()

    batch_size    = args.batch_size
    accumulate    = args.accumulation_steps
    is_baseline   = args.baseline
    model         = args.base_models
    learning_rate = args.learning_rate


    lora_config = LoraConfig(task_type="CAUSAL_LM", inference_mode=False, r=32, lora_alpha=64, lora_dropout=0.1)
    dataset = BlendShapeDataset("data", "SONA", 600, batch_size, False, model, is_baseline)
    if is_baseline:
        model = Baseline(learning_rate, model, lora_config)
        
        summary = ModelSummary(max_depth=2)
        logger = TensorBoardLogger("logs", name="Uncertain-Agents-Baseline")
        checkpoint = ModelCheckpoint(dirpath="baseline-model", filename='cues-{epoch}-{val_loss:.2f}-{val_kl:.2f}-{val_rec:.2f}-{val_meta:.2f}', save_top_k=2, monitor="val_loss")
        checkpoint_periodic = ModelCheckpoint(dirpath="baseline-model", filename='cues-{epoch}-{val_loss:.2f}-{val_kl:.2f}-{val_rec:.2f}-{val_meta:.2f}', every_n_epochs=250, save_top_k=-1)

        auto_wrap_policy = {LlamaDecoderLayer}
        trainer = pl.Trainer(logger=logger, max_epochs=1000, precision='16-mixed',
                    accumulate_grad_batches=accumulate, accelerator="cuda",
                    strategy=FSDPStrategy(auto_wrap_policy=auto_wrap_policy), devices=2,
                    callbacks=[checkpoint, checkpoint_periodic, summary]) 
    else:
        model = BlendShapeVAE(len(ROWS), 32, 114 * 16, 2, 114, learning_rate, "add", model, 4096, lora_config)
        logger = TensorBoardLogger("logs", name="Uncertain-Agents-GPT")
        checkpoint = ModelCheckpoint(dirpath="gpt-gold-model", filename='cues-{epoch}-{val_loss:.2f}-{val_kl:.2f}-{val_rec:.2f}-{val_meta:.2f}', save_top_k=2, monitor="val_loss")
        checkpoint_periodic = ModelCheckpoint(dirpath="gpt-gold-model", filename='cues-{epoch}-{val_loss:.2f}-{val_kl:.2f}-{val_rec:.2f}-{val_meta:.2f}', every_n_epochs=250, save_top_k=-1)
        summary = ModelSummary(max_depth=2)

        trainer = pl.Trainer(logger=logger, max_epochs=1000, precision='16-mixed',
                            accumulate_grad_batches=accumulate, gradient_clip_val=5.0,
                            strategy="ddp_find_unused_parameters_true",
                            callbacks=[checkpoint, checkpoint_periodic, summary]) 
    trainer.fit(model=model, train_dataloaders=dataset)
