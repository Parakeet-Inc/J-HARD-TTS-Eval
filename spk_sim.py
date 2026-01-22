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

SAMPLING_RATE = 16000
CONFIG_PATH = Path("./config.yaml")
EPS = 1e-7

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

        # if the length of the audio is less than 3 seconds, skip
        if wav.shape[-1] / SAMPLING_RATE < 3.0:
            return None

        # return speaker embedding
        return self.ecapa_tdnn(torch.from_numpy(wav).cuda()).cpu()


def main():
    # load config
    cfg = OmegaConf.load(CONFIG_PATH)

    # load model
    model = WavLMLargeECAPATDNN(cfg.wavlm_large_ecapa_tdnn_model_path)

    # process each subset
    all_spk_sims = []
    for subset_name in cfg.subsets:
        print(f"Processing subset: {subset_name}")

        spk_sims = []

        dataset = load_dataset(
            "Parakeet-Inc/J-HARD-TTS-Eval", subset_name, split="test"
        )
        for data in tqdm(dataset):
            wav_prompt = torch.from_numpy(data["prompt_audio"]["array"])
            if wav_prompt.dim() == 1:
                wav_prompt = wav_prompt.unsqueeze(0)
            fs_prompt = data["prompt_audio"]["sampling_rate"]
            tts_file_name = data["id"]

            for tts_speech_path in (Path(cfg.wav_dir_path_root) / subset_name).glob(
                f"{tts_file_name.split('.')[0]}*.wav"
            ):
                # load audio files
                wav_tts, fs_tts = torchaudio.load(tts_speech_path)

                # extract speaker embeddings
                spk_emb_prompt = model(wav_prompt, fs_prompt)
                spk_emb_tts = model(wav_tts, fs_tts)
                if spk_emb_tts is None:
                    continue

                # speaker similarity
                spk_sim = torch.nn.functional.cosine_similarity(
                    spk_emb_prompt, spk_emb_tts, dim=1
                )

                # store the cosine similarity
                # NOTE: only store if the cosine similarity is greater than 0.1,
                # this is because if the speaker similarity is too low,
                # there is a possibility that the generated speech may contain much silence,
                # and then this sample is eliminated to ensure proper evaluation.
                if spk_sim > 0.1:
                    spk_sims.append(spk_sim.item())

        # output subset average cosine similarity
        with open(
            Path(Path(cfg.result_dir_path_root) / subset_name / "spk_sim.txt"), "w"
        ) as f:
            f.write(f"{np.mean(spk_sims):#.4g}")

        all_spk_sims.extend(spk_sims)

    # output subset average cosine similarity
    with open(Path(Path(cfg.result_dir_path_root) / "spk_sim_overall.txt"), "w") as f:
        f.write(f"{np.mean(all_spk_sims):#.4g}\n")


if __name__ == "__main__":
    main()
