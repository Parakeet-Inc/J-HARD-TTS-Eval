import json
import warnings
from pathlib import Path
from typing import List

import librosa
import reazonspeech.espnet.asr as espnet_asr
import torch
import torchaudio
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

warnings.filterwarnings(
    "ignore", message=".*torch.cuda.amp.autocast.*", category=FutureWarning
)

warnings.filterwarnings(
    "ignore", message=".*stft with return_complex=False.*", category=UserWarning
)


SAMPLING_RATE = 16000
CONFIG_PATH = Path("./config.yaml")
EPS = 1e-7


class AudioDataset(Dataset):
    """
    dataset class for parallel loading of audio files.
    """

    def __init__(
        self,
        corpora_path: Path,
        wav_dir_path: Path,
        n_times: int = 5,
    ):
        """
        Args:
            corpora_path (Path): Path to the corpora text file.
            wav_dir_path (Path): Path to the directory containing wav files.
            n_times (int): Number of times each sample is generated (default: 5).
        """
        self.wav_dir_path = wav_dir_path
        self.n_times = n_times

        # load subset
        self.subset_data = []
        with open(corpora_path, "r") as f:
            lines = f.read().splitlines()
        for line in lines:
            _, _, file_name, text = line.split("|")
            self.subset_data.append([file_name, text])

    def __len__(self):
        """
        return the number of samples in the dataset.
        """
        return len(self.subset_data)

    def __getitem__(self, idx: int) -> dict:
        """
        load n_times wav files for the given index.

        Args:
            idx (int): Index of the sample to load.

        Returns:
            dict: A dictionary containing:
                - "file_name": The file name of the sample.
                - "text": The transcription text of the sample.
                - "wavs": A list of loaded waveforms (each is a 1D Tensor).
        """
        file_name, text = self.subset_data[idx]
        wavs = []

        # load n_times wav files
        for i in range(self.n_times):
            wav_path = self.wav_dir_path / (file_name.split(".")[0] + f"-{i}.wav")
            assert wav_path.exists(), f"File not found: {wav_path}"

            # Load wav
            wav, fs = torchaudio.load(wav_path)
            wav = wav.mean(dim=0)  # to mono

            # resample by soxr_vhq to 16k if needed
            if fs != SAMPLING_RATE:
                wav_resampled = librosa.resample(
                    wav.numpy(),
                    orig_sr=fs,
                    target_sr=SAMPLING_RATE,
                    res_type="soxr_vhq",
                )
                wav = torch.from_numpy(wav_resampled)

            # normalization
            wav = wav / (torch.abs(wav).max() + EPS) * 0.8

            wavs.append(wav)

        return {
            "file_name": file_name,
            "text": text,
            "wavs": wavs,
        }


class WhisperASR:
    """
    Whisper ASR model class.
    batch inference using Whisper-large-v3 model.
    """

    def __init__(self):
        model_id = "openai/whisper-large-v3"
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = (
            WhisperForConditionalGeneration.from_pretrained(
                model_id, dtype=torch.float16
            )
            .eval()
            .to("cuda")
        )

    @torch.inference_mode()
    def asr_batch(self, wavs: List[torch.Tensor]):
        """
        batch inference using Whisper model.

        Args:
            wavs (List[torch.Tensor]): List of waveforms (each is a 1D Tensor).

        Returns:
            List of transcribed texts.
        """
        inputs = self.processor(
            [wav.numpy() for wav in wavs],
            sampling_rate=SAMPLING_RATE,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        predicted_ids = self.model.generate(
            inputs.input_features.to("cuda", dtype=torch.float16),
            attention_mask=inputs.attention_mask.to("cuda"),
            language="ja",
            task="transcribe",
            use_cache=True,
        )
        hypos = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return hypos


class ReazonSpeechESPNetv2ASR:
    """
    ReazonSpeech ESPNetv2 ASR model class.
    pseudo batch inference using ReazonSpeech ESPNetv2 ASR model.
    """

    def __init__(self):
        self.reazonspeech_espnet = espnet_asr.load_model(device="cuda")

    def asr_batch(self, wavs: List[torch.Tensor]):
        """
        batch inference using ReazonSpeech ESPNetv2 ASR model.
        NOTE: This implementation processes one by one due to model interface limitations.

        Args:
            wavs (List[torch.Tensor]): List of waveforms (each is a 1D Tensor).

        Returns:
            List of transcribed texts.
        """
        hypos = []
        for wav in wavs:
            wav = espnet_asr.interface.AudioData(wav.numpy(), SAMPLING_RATE)
            hypo = espnet_asr.transcribe(
                self.reazonspeech_espnet,
                wav,
                config=espnet_asr.TranscribeConfig(verbose=False),
            ).text
            hypos.append(hypo)
        return hypos


def main():
    # load config
    cfg = OmegaConf.load(CONFIG_PATH)

    # load asr models
    print("Loading ASR models...")
    whisper_asr = WhisperASR()
    reazon_speech_asr = ReazonSpeechESPNetv2ASR()
    print("ASR models loaded.")

    # process each subset
    ids = [i for i in range(cfg.n_times_per_sample)]
    for subset_name in cfg.subsets:
        print(f"Processing subset: {subset_name}")
        corpora_path = Path(cfg.corpora_dir_path) / f"{subset_name}.txt"
        wav_dir_path = Path(cfg.wav_dir_path_root) / subset_name

        # load dataset and dataloader
        dataset = AudioDataset(
            corpora_path, wav_dir_path, n_times=cfg.n_times_per_sample
        )
        dataloader = DataLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=4,
        )

        # main processing
        outputs = []
        for batch in tqdm(dataloader):
            whisper_hypos = whisper_asr.asr_batch(batch["wavs"])
            reazon_hypos = reazon_speech_asr.asr_batch(batch["wavs"])
            outputs.append(
                {
                    "file_name": batch["file_name"],
                    "text": batch["text"],
                    "hypo_whisper": dict(zip(ids, whisper_hypos)),
                    "hypo_reazon": dict(zip(ids, reazon_hypos)),
                }
            )

        # write to JSONL file
        output_file_path = Path(cfg.result_dir_path_root) / subset_name / "asr.jsonl"
        with open(output_file_path, "w") as f:
            for item in outputs:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
