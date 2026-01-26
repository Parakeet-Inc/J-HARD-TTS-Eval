import json
import sys
import types
import warnings
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio
from datasets import load_dataset
from ecapa_tdnn import ECAPA_TDNN_SMALL
from omegaconf import OmegaConf
from tqdm import tqdm

if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda backend: None

if "torchaudio.sox_effects" not in sys.modules:
    dummy_sox = types.ModuleType("torchaudio.sox_effects")
    dummy_sox.apply_effects_tensor = lambda *args, **kwargs: (None, None)
    sys.modules["torchaudio.sox_effects"] = dummy_sox
    torchaudio.sox_effects = dummy_sox

warnings.filterwarnings(
    "ignore",
    message="Support for mismatched key_padding_mask and attn_mask is deprecated",
)
warnings.filterwarnings(
    "ignore",
    message="`torch.nn.utils.weight_norm` is deprecated",
    category=FutureWarning,
)


SAMPLING_RATE = 16000
CONFIG_PATH = Path("./config.yaml")
EPS = 1e-7


class WavLMLargeECAPATDNN(torch.nn.Module):
    """
    WavLM Large + ECAPA-TDNN speaker embedding extractor.
    """

    def __init__(self, model_path: str | Path):
        """
        Args:
            model_path (str | Path): Path to the pretrained model checkpoint.
        """
        super().__init__()
        self.ecapa_tdnn = ECAPA_TDNN_SMALL(
            feat_dim=1024, feat_type="wavlm_large", config_path=None
        )
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.ecapa_tdnn.load_state_dict(state_dict["model"], strict=False)
        self.ecapa_tdnn.eval().cuda()

    @torch.inference_mode()
    def forward(self, wav: torch.Tensor, fs: int) -> torch.Tensor:
        """
        Args:
            wav (torch.Tensor): Waveform tensor (shape: [num_channels, num_samples]).
            fs (int): Sampling rate of the input waveform.

        Returns:
            torch.Tensor: Speaker embedding tensor (shape: [1, embedding_dim]).
        """
        # to mono
        wav = wav.mean(dim=0, keepdim=True)

        # Resample by soxr_vhq to 16k if needed
        if fs != SAMPLING_RATE:
            wav = librosa.resample(
                wav.numpy(),
                orig_sr=fs,
                target_sr=SAMPLING_RATE,
                res_type="soxr_vhq",
            )
            wav = torch.from_numpy(wav)

        # normalization
        wav = wav / (torch.abs(wav).max() + EPS) * 0.8

        # trim silence
        wav, _ = librosa.effects.trim(
            wav.numpy(), top_db=40, frame_length=512, hop_length=128
        )

        # Short audio (< 2s) makes speaker estimation inherently difficult,
        # and the model was likely not sufficiently trained on such short data.
        # Therefore, skip such samples to avoid unreliable embeddings.
        if wav.shape[-1] / SAMPLING_RATE < 2.0:
            return None

        # wavlm was likely pre-trained without long sequences, so performance may drop on long audio.
        # trim to 20s to avoid this degradation and extract embeddings correctly.
        if wav.shape[-1] / SAMPLING_RATE > 20.0:
            wav = wav[:, : SAMPLING_RATE * 20]

        # return speaker embedding
        return self.ecapa_tdnn(torch.from_numpy(wav).cuda()).cpu()


def main():
    # load config
    cfg = OmegaConf.load(CONFIG_PATH)

    # load model
    model = WavLMLargeECAPATDNN(cfg.wavlm_large_ecapa_tdnn_model_path)

    # make spk_sim lists
    spk_sims = []
    spk_sims_cer_0_filtered = []
    spk_sims_cer_below_10_filtered = []
    spk_sims_cer_below_30_filtered = []
    spk_sims_cer_below_50_filtered = []
    spk_sims_cer_below_100_filtered = []

    # process each subset
    for subset_name in cfg.subsets:
        print(f"Processing subset: {subset_name}")

        # load cer result
        with open(Path(cfg.result_dir_path_root) / subset_name / "cer.jsonl", "r") as f:
            jsonl_data = [json.loads(line) for line in f.read().splitlines()]
        cer_result = {item["file_name"]: item["cer"] for item in jsonl_data}

        spk_sims_ = []
        spk_sims_cer_0_filtered_ = []
        spk_sims_cer_below_10_filtered_ = []
        spk_sims_cer_below_30_filtered_ = []
        spk_sims_cer_below_50_filtered_ = []
        spk_sims_cer_below_100_filtered_ = []

        dataset = load_dataset(
            "Parakeet-Inc/J-HARD-TTS-Eval", subset_name, split="test"
        )
        for data in tqdm(dataset):
            wav_prompt = torch.from_numpy(data["prompt_audio"]["array"])
            if wav_prompt.dim() == 1:
                wav_prompt = wav_prompt.unsqueeze(0)
            fs_prompt = data["prompt_audio"]["sampling_rate"]
            tts_file_name = data["id"]

            for i in range(cfg.n_times_per_sample):
                # load audio files
                wav_tts, fs_tts = torchaudio.load(
                    Path(cfg.wav_dir_path_root)
                    / subset_name
                    / (tts_file_name + f"-{i}.wav")
                )

                # extract speaker embeddings
                spk_emb_tts = model(wav_tts, fs_tts)
                if spk_emb_tts is None:
                    continue
                spk_emb_prompt = model(wav_prompt, fs_prompt)

                # speaker similarity
                spk_sim = torch.nn.functional.cosine_similarity(
                    spk_emb_prompt, spk_emb_tts, dim=1
                )
                spk_sims_.append(spk_sim.item())
                if cer_result[tts_file_name + ".wav"][str(i)] == 0.0:
                    spk_sims_cer_0_filtered_.append(spk_sim.item())
                if cer_result[tts_file_name + ".wav"][str(i)] <= 10.0:
                    spk_sims_cer_below_10_filtered_.append(spk_sim.item())
                if cer_result[tts_file_name + ".wav"][str(i)] <= 30.0:
                    spk_sims_cer_below_30_filtered_.append(spk_sim.item())
                if cer_result[tts_file_name + ".wav"][str(i)] <= 50.0:
                    spk_sims_cer_below_50_filtered_.append(spk_sim.item())
                if cer_result[tts_file_name + ".wav"][str(i)] <= 100.0:
                    spk_sims_cer_below_100_filtered_.append(spk_sim.item())

        # output subset average cosine similarity
        with open(
            Path(Path(cfg.result_dir_path_root) / subset_name / "spk_sim.txt"), "w"
        ) as f:
            if len(spk_sims_) == 0:
                f.write("spk_sim (no filterd): N/A\n")
            else:
                f.write(f"spk_sim (no filterd): {np.mean(spk_sims_):#.4g}\n")
            if len(spk_sims_cer_0_filtered_) == 0:
                f.write("spk_sim (cer == 0 filtered): N/A\n")
            else:
                f.write(
                    f"spk_sim (cer == 0 filtered): {np.mean(spk_sims_cer_0_filtered_):#.4g}\n"
                )
            if len(spk_sims_cer_below_10_filtered_) == 0:
                f.write("spk_sim (cer <= 10 filtered): N/A\n")
            else:
                f.write(
                    f"spk_sim (cer <= 10 filtered): {np.mean(spk_sims_cer_below_10_filtered_):#.4g}\n"
                )
            if len(spk_sims_cer_below_30_filtered_) == 0:
                f.write("spk_sim (cer <= 30 filtered): N/A\n")
            else:
                f.write(
                    f"spk_sim (cer <= 30 filtered): {np.mean(spk_sims_cer_below_30_filtered_):#.4g}\n"
                )
            if len(spk_sims_cer_below_50_filtered_) == 0:
                f.write("spk_sim (cer <= 50 filtered): N/A\n")
            else:
                f.write(
                    f"spk_sim (cer <= 50 filtered): {np.mean(spk_sims_cer_below_50_filtered_):#.4g}\n"
                )
            if len(spk_sims_cer_below_100_filtered_) == 0:
                f.write("spk_sim (cer <= 100 filtered): N/A\n")
            else:
                f.write(
                    f"spk_sim (cer <= 100 filtered): {np.mean(spk_sims_cer_below_100_filtered_):#.4g}\n"
                )

        spk_sims.extend(spk_sims_)
        spk_sims_cer_0_filtered.extend(spk_sims_cer_0_filtered_)
        spk_sims_cer_below_10_filtered.extend(spk_sims_cer_below_10_filtered_)
        spk_sims_cer_below_30_filtered.extend(spk_sims_cer_below_30_filtered_)
        spk_sims_cer_below_50_filtered.extend(spk_sims_cer_below_50_filtered_)
        spk_sims_cer_below_100_filtered.extend(spk_sims_cer_below_100_filtered_)

    # output subset average cosine similarity
    with open(Path(Path(cfg.result_dir_path_root) / "spk_sim_overall.txt"), "w") as f:
        f.write(f"spk_sim (no filterd): {np.mean(spk_sims):#.4g}\n")
        f.write(
            f"spk_sim (cer == 0 filtered): {np.mean(spk_sims_cer_0_filtered):#.4g}\n"
        )
        f.write(
            f"spk_sim (cer <= 10 filtered): {np.mean(spk_sims_cer_below_10_filtered):#.4g}\n"
        )
        f.write(
            f"spk_sim (cer <= 30 filtered): {np.mean(spk_sims_cer_below_30_filtered):#.4g}\n"
        )
        f.write(
            f"spk_sim (cer <= 50 filtered): {np.mean(spk_sims_cer_below_50_filtered):#.4g}\n"
        )
        f.write(
            f"spk_sim (cer <= 100 filtered): {np.mean(spk_sims_cer_below_100_filtered):#.4g}\n"
        )


if __name__ == "__main__":
    main()
