from pathlib import Path

import click
import hydra
import numpy as np
import soundfile as sf
import torch
import torchaudio
from hydra import compose, initialize
from hydra.utils import instantiate
from lightning import LightningModule
from loguru import logger
from omegaconf import OmegaConf

from fish_speech.utils.file import AUDIO_EXTENSIONS

# 注册一个解析器 eval，用于处理配置文件中的动态计算。
OmegaConf.register_new_resolver("eval", eval)


def load_model(config_name, checkpoint_path, device="cuda"):
    """
    加载预训练模型
    :param config_name: 配置文件名称
    :param checkpoint_path: 模型检查点路径
    :param device: 运行设备，默认是CUDA
    :return: 加载的模型
    """

    # 清除当前的 Hydra 实例
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    # 初始化 Hydra
    with initialize(version_base="1.3", config_path="../../fish_speech/configs"):
        cfg = compose(config_name=config_name)

    # 根据配置实例化模型
    model: LightningModule = instantiate(cfg.model)

    #  从检查点加载模型状态字典
    state_dict = torch.load(
        checkpoint_path,
        map_location=model.device,
    )

    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict, strict=False)

    # 设置模型为评估模式
    model.eval()
    model.to(device)
    logger.info("Restored model from checkpoint")

    return model


@torch.no_grad()
@click.command()
@click.option(
    "--input-path",
    "-i",
    default="test.wav",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--output-path", "-o", default="fake.wav", type=click.Path(path_type=Path)
)
@click.option("--config-name", "-cfg", default="vqgan_pretrain")
@click.option(
    "--checkpoint-path",
    "-ckpt",
    default="checkpoints/vq-gan-group-fsq-2x1024.pth",
)
@click.option(
    "--device",
    "-d",
    default="cuda",
)
def main(input_path, output_path, config_name, checkpoint_path, device):
    """
    主函数，处理音频文件
    :param input_path: 输入音频文件路径
    :param output_path: 输出音频文件路径
    :param config_name: 配置文件名称
    :param checkpoint_path: 模型检查点路径
    :param device: 运行设备
    """

    model = load_model(config_name, checkpoint_path, device=device)

    if input_path.suffix in AUDIO_EXTENSIONS:
        logger.info(f"Processing in-place reconstruction of {input_path}")

        # Load audio
        audio, sr = torchaudio.load(str(input_path))
        if audio.shape[0] > 1:
            # 如果是立体声，则转换为单声道
            audio = audio.mean(0, keepdim=True)

        # 重采样音频
        audio = torchaudio.functional.resample(audio, sr, model.sampling_rate)

        # 将音频数据发送到模型设备上
        audios = audio[None].to(model.device)
        logger.info(
            f"Loaded audio with {audios.shape[2] / model.sampling_rate:.2f} seconds"
        )

        # VQ Encoder 通过模型进行编码，生成编码索引
        audio_lengths = torch.tensor(
            [audios.shape[2]], device=model.device, dtype=torch.long
        )
        indices = model.encode(audios, audio_lengths)[0][0]

        logger.info(f"Generated indices of shape {indices.shape}")

        # 保存编码索引
        np.save(output_path.with_suffix(".npy"), indices.cpu().numpy())
    elif input_path.suffix == ".npy":
        logger.info(f"Processing precomputed indices from {input_path}")

        # 如果输入文件是 .npy 文件，加载预计算的编码索引，并进行验证。
        indices = np.load(input_path)
        indices = torch.from_numpy(indices).to(model.device).long()
        assert indices.ndim == 2, f"Expected 2D indices, got {indices.ndim}"
    else:
        raise ValueError(f"Unknown input type: {input_path}")

    # 通过模型解码生成伪音频
    feature_lengths = torch.tensor([indices.shape[1]], device=model.device)
    fake_audios = model.decode(
        indices=indices[None], feature_lengths=feature_lengths, return_audios=True
    )

    # 计算生成音频的时长
    audio_time = fake_audios.shape[-1] / model.sampling_rate

    logger.info(
        f"Generated audio of shape {fake_audios.shape}, equivalent to {audio_time:.2f} seconds from {indices.shape[1]} features, features/second: {indices.shape[1] / audio_time:.2f}"
    )

    # 保存解码后的音频
    fake_audio = fake_audios[0, 0].float().cpu().numpy()
    sf.write(output_path, fake_audio, model.sampling_rate)
    logger.info(f"Saved audio to {output_path}")


if __name__ == "__main__":
    main()
