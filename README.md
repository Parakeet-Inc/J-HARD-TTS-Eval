# J-HARD-TTS-Eval

## Paper
[系列整合性評価に特化した高難度日本語テキスト音声合成コーパスの検討 (A Study on a High-Difficulty Japanese Text-to-Speech Corpus Specialized for Sequence Consistency Evaluation)](https://drive.google.com/file/d/1CZ74Nvmca7nneK2zCfRyogFAMVLUx0Ik/view?usp=drive_link) (Japanese article)

Note: Results in this README may differ slightly from the paper due to refactoring performed during the code release process.


## Overview
This repository provides the evaluation code for **J-HARD-TTS-Eval**, a benchmark designed to evaluate the robustness of **autoregressive Japanese Text-To-Speech(TTS) models**. This benchmark consists of the following 4 subsets, each designed to assess model robustness from a different perspective.

| Subset | Evaluation Objective |
| :--- | :--- |
| **Short** | Composed of extremely short utterances containing only 1 to 3 moras. Evaluates the model's stability when synthesizing very short sequences. |
| **Repetition** | Composed of sentences containing frequent repetitions of the same phrase. Evaluates robustness against skipping or additional repetition, and checks if the model can complete the utterance without premature stopping. |
| **Rhyme** | Composed of sentences rich in alliteration or rhyme. Evaluates similar aspects to the Repetition subset, but tests stability against recurring phonological patterns rather than identical lexical repetition. |
| **Continuation** | Composed of incomplete sentences where the context is abruptly cut off. Evaluates whether the model can synthesize faithfully to the input text without over-completing the context or arbitrarily generating continuations not present in the input text. |

For more details on each subset, please refer to the `./corpora` directory.
Each line in the files contains data in the following format:

```txt
[prompt speech file name] | [prompt speech transcript] | [target file name] | [target text]
```

To focus on the robustness of the TTS model itself, target text is designed to minimize the impact of whether G2P (Grapheme-to-Phoneme) is used or not, as well as its potential performance discrepancies.
Specifically, we excluded words with ambiguous readings, such as **"今日"** (which can be read as *Kyou* or *Konnichi*) and **"17"** (*Juushichi* or *Juunana*).
Additionally, the Kanji characters used are strictly limited to the scope of **Joyo Kanji** (常用漢字; Japanese regular-use characters).

Each utterance is accompanied by prompt audio and a transcript for zero-shot synthesis.
Following the approach of the prior work [Seed-TTS-Eval](https://github.com/BytedanceSpeech/seed-tts-eval) [1], these were sourced from the [Common Voice](https://commonvoice.mozilla.org/en) dataset ([Hugging Face link](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0)).
Since Common Voice contains many recordings with poor recording environments or short durations, we selected prompt candidates by filtering for audio that has a speech duration of 5 seconds or longer after removing leading and trailing silence, and a UTMOS [2] score of 2.5 or higher. From these candidates, the final prompts were randomly selected.
The data can be accessed using the Hugging Face Datasets library as described below, but you can download in bulk from [this Google Drive link](https://drive.google.com/file/d/1pcnRvqgNFcyGk0RzeWoxAXrCaQRCukOc/view?usp=drive_link).

# Install

```sh
# clone this repo
$ git clone git@github.com:Parakeet-Inc/J-HARD-TTS-Eval.git

# optional: if uv is not installed
$ pip install uv

# create virtual environment and install dependencies
$ uv sync

# download speaker embedding model (WavLM-Large ECAPA-TDNN)
$ gdown --fuzzy https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view?pli=1
```


# Evaluation

## 1. TTS data preparation
Synthesize speech from the text for each subset using the TTS model you want to evaluate.
The required data can be accessed using the Hugging Face Datasets library.

```python
from datasets import load_dataset

subset = "rhyme"  # "short", "repetition", "rhyme", "continuation"

ds = load_dataset("Parakeet-Inc/J-HARD-TTS-Eval", subset, split="test")

sample = ds[0]
audio_data = sample["prompt_audio"]

print(
    f"""
    file name: {sample["id"]}
    prompt audio: {audio_data["array"]}
    prompt audio data type: {type(audio_data["array"])}, sampling rate: {audio_data["sampling_rate"]}
    prompt text: {sample["prompt_text"]}
    target text: {sample["text"]}
    """
)
"""
>> file name: rhyme-01
>> prompt audio: [ 0.0000000e+00  0.0000000e+00  0.0000000e+00 ... -3.7011851e-06
>> -3.8374778e-06  7.2635004e-07]
>> prompt audio data type: <class 'numpy.ndarray'>, sampling rate: 24000
>> prompt text: 私は宿へ引っ張ってゆかれましたが、門口には、番人がちゃんと一人立っています。しかし、庭の中を歩きまわることだけは許されました。
>> target text: 同僚と同様に、独自の努力で土台を築く同級生の堂々とした度胸に、ドリーはどうしようもなく動揺した。
"""
```
Perform inference 5 times for each target file (reasons detailed below) and save the results in the following directory structure:

```text
tts_data/
├── short/
│   ├── short-01-0.wav
│   ├── short-01-1.wav
│   ├── short-01-2.wav
│   ├── short-01-3.wav
│   ├── short-01-4.wav
│   ├── short-02-0.wav
│   ├── short-02-1.wav
│   ├── ...
│   └── short-40-4.wav
├── repetition/
│   ├── repetition-01-0.wav
│   ├── ...
│   └── repetition-40-4.wav
├── rhyme/
│   ├── rhyme-01-0.wav
│   ├── ...
│   └── rhyme-40-4.wav
└── continuation/
    ├── continuation-01-0.wav
    ├── ...
    └── continuation-40-4.wav
```


## 2. Character Error Rate (CER)
We calculate the Character Error Rate (CER) using a combination of the [Whisper large-v3](https://huggingface.co/openai/whisper-large-v3) [3] model and the [ReazonSpeech ESPNet v2](https://huggingface.co/reazon-research/reazonspeech-espnet-v2) model.

While Whisper large-v3 was adopted for English transcription in the prior work Seed-TTS-Eval, we observed that its Encoder-Decoder architecture makes it particularly prone to hallucinations in the **Repetition** subset, which consists of repetitive text.
Furthermore, in the **Short** subset, composed of extremely short audio, it often incorrectly outputs phrases like "ご視聴ありがとうございました"("Thank you for watching"), likely due to the influence of its training data from YouTube.
Preliminary experiments confirmed that the ReazonSpeech ESPNet v2 model performs robustly in these cases.
Therefore, for all audio evaluations, we perform recognition using both Whisper large-v3 and ReazonSpeech ESPNet v2, adopting the result with the lower CER for each utterance.
Additionally, since recent autoregressive models typically involve sampling during inference, results vary across runs.
A robust TTS method is expected to generate stable output across multiple inferences with the same input.

To evaluate this, we perform inference 5 times for each utterance.
From these 5 inferences, we calculate the following metrics:

* **CER_Best**: CER calculated using only the transcription with the lowest error rate among the 5 runs.
* **CER_Worst**: CER calculated using only the transcription with the highest error rate.
* **CER_Average**: The average of CERs calculated from all transcriptions.

Typically, CER_Average is used as the primary metric. However, examining the gap between CER_Best and CER_Worst allows for assessing the model's robustness to sampling variability.
Note that while Seed-TTS-Eval used Macro CER, we calculate **Micro CER** in this benchmark to account for the variance in character counts across utterances within each subset.

First, run the following command to perform ASR:
```sh
$ uv run asr.py
```
The ASR results will be saved as asr.jsonl within each subset directory under result/.

Next, calculate the CER. For CER calculation, the text is first normalized and then converted into Kana (phonetic characters) using [pyopenjtalk](https://github.com/r9y9/pyopenjtalk).
This is because the benchmark focuses on the TTS model's robustness in correctly pronouncing the content.
Converting to Kana allows us to uniformly handle cases where the ASR model transcribes the speech into different Kanji characters that share the same reading (homophones).
```sh
$ uv run cer.py
```
This will save the CER results as cer.txt within each subset directory under result/. Additionally, the CER for each file and its each 5 run is saved as cer.jsonl. This file will be used in the subsequent Speaker Similarity evaluation.


## 3. Speaker similarity
We use [WavLM-Large [4] ECAPA-TDNN [5]](https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view?pli=1) to calculate the Speaker Similarity between the prompt speech and the zero-shot synthesized speech, following the approach of Seed-TTS-Eval.
To examine the impact of content accuracy on Speaker Similarity scores, we calculate scores by filtering data based on CER thresholds (`0`, `10`, `30`, `50`, `100` and `no filtered`).
We apply the following preprocessing and filtering to the audio:

- **Maximum Length Limit (20s)**:
    Taking into account the training data distribution of the WavLM model, audio exceeding 20 seconds is truncated to 20 seconds. This prevents performance degradation caused by inputs of unexpected lengths.
- **Minimum Length Limit (2s)**:
    Extracting accurate speaker embeddings from extremely short speech is difficult. Additionally, when synthesis fails in the **Short** subset, the output often consists only of silence or noise; since these lack speaker characteristics, they are unsuitable for evaluation. Therefore, speech clips shorter than 2 seconds (after trimming leading and trailing silence) are excluded from the calculation (Note: This results in most data in the **Short** subset being skipped).

Run the following command to calculate Speaker Similarity:
```sh
$ uv run spk_sim.py
```
The Speaker Similarity results for each subset will be saved as spk_sim.txt within each subset directory under result/. Additionally, the aggregated results for all utterances will be saved as results/spk_sim_overall.txt.

Once all evaluations are complete, the result/ directory will be structured as follows:
```text
result/
├── <Subset Name>/  # (short, repetition, rhyme, continuation)
│   ├── asr.jsonl
│   ├── cer.jsonl
│   ├── cer.txt
│   └── spk_sim.txt
└── spk_sim_overall.txt
```


# Model list
We present the evaluation results for the following TTS models that support Japanese.
In addition to recent LM-based zero-shot models, we also evaluated conventional autoregressive methods (Transformer-TTS, Tacotron2) and non-autoregressive methods (FastSpeech2) trained on the [JSUT corpus](https://sites.google.com/site/shinnosuketakamichi/publication/jsut) [6] for reference.

| Task | Model | Release Date | AR? | # Params | Paper Link |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Zero-shot | [XTTS-v2](https://huggingface.co/coqui/XTTS-v2) | 2023-12 | Yes | 441.0M (424.2M) | [XTTS: a Massively Multilingual Zero-Shot Text-to-Speech Model](https://arxiv.org/abs/2406.04904) |
|  | [CosyVoice2-0.5B](https://github.com/FunAudioLLM/CosyVoice) | 2024-05 | Yes | AR: 505.8M (357.9M)<br>NAR: 112.5M | [CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models](https://arxiv.org/abs/2412.10117) |
|  | [FishAudio-S1-mini](https://huggingface.co/fishaudio/openaudio-s1-mini)<br>(OpenAudio S1-mini) | 2025-05 | Yes | AR: 801.4M (440.5M)<br>NAR: 58.73M | [Fish-Speech: Leveraging Large Language Models for Advanced Multilingual Text-to-Speech Synthesis](https://arxiv.org/abs/2411.01156) |
|  | [Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) | 2026-01 | Yes | AR: 764.2M (437.3M)<br>NAR: 141.6M | [Qwen3-TTS Technical Report](https://arxiv.org/abs/2601.15621) |
|  | [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | 2026-01 | Yes | AR: 1.703B (1.403B)<br>NAR: 175.1M | [Qwen3-TTS Technical Report](https://arxiv.org/abs/2601.15621) |
| Single Speaker (JSUT) | [Tacotron2](https://huggingface.co/espnet/kan-bayashi_jsut_tacotron2_accent_with_pause) | - | Yes | 26.66M | [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/abs/1712.05884) |
|  | [Transformer-TTS](https://huggingface.co/espnet/kan-bayashi_jsut_transformer_accent_with_pause) | - | Yes | 33.04M | [Neural Speech Synthesis with Transformer Network](https://arxiv.org/abs/1809.08895) |
|  | [FastSpeech2](https://huggingface.co/espnet/kan-bayashi_jsut_fastspeech2_accent_with_pause) | - | No | 37.12M | [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558) |



# Results
## CER
### Subset: Short
| Task | Model | Best [%] | Average [%] | Worst [%] |
| :--- | :--- | :--- | :--- | :--- |
| Zero-shot | XTTS-v2 | 5.512 | 14.33 | 31.50 |
|  | CosyVoice2-0.5B | 22.83 | 71.50 | 123.6 |
|  | FishAudio-S1-mini | **0.7874** | 15.59 | 48.82 |
|  | Qwen3-TTS-12Hz-0.6B-Base | 7.087 | 22.36 | 45.67 |
|  | Qwen3-TTS-12Hz-1.7B-Base | 1.575 | **4.724** | **11.02** |
| Single Speaker (JSUT) | Tacotron2 | - | 5.512 | - |
|  | Transformer-TTS | - | 115.7 | - |
|  | FastSpeech2 | - | 9.449 | - |

### Subset: Repetition
| Task | Model | Best [%] | Average [%] | Worst [%] |
| :--- | :--- | :--- | :--- | :--- |
| Zero-shot | XTTS-v2 | 7.792 | 12.12 | 18.61 |
|  | CosyVoice2-0.5B | 8.139 | 15.25 | 28.68 |
|  | FishAudio-S1-mini | 11.81 | 35.19 | 79.90 |
|  | Qwen3-TTS-12Hz-0.6B-Base | 6.799 | 13.01 | 21.49 |
|  | Qwen3-TTS-12Hz-1.7B-Base | **5.261** | **10.57** | **17.67** |
| Single speaker (JSUT) | Tacotron2 | - | 7.940 | - |
|  | Transformer-TTS | - | 15.24 | - |
|  | FastSpeech2 | - | 3.623 | - |

### Subset: Rhyme
| Task | Model | Best [%] | Average [%] | Worst [%] |
| :--- | :--- | :--- | :--- | :--- |
| Zero-shot | XTTS-v2 | **0.1419** | **1.064** | 3.122 |
|  | CosyVoice2-0.5B | 0.1774 | 1.398 | 4.576 |
|  | FishAudio-S1-mini | 0.4966 | 1.313 | **3.015** |
|  | Qwen3-TTS-12Hz-0.6B-Base | 2.128 | 4.292 | 7.627 |
|  | Qwen3-TTS-12Hz-1.7B-Base | 0.6031 | 2.469 | 4.753 |
| Single speaker (JSUT) | Tacotron2 | - | 0.07095 | - |
|  | Transformer-TTS | - | 3.086 | - |
|  | FastSpeech2 | - | 0.0 | - |

### Subset: Continuation
| Task | Model | Best [%] | Average [%] | Worst [%] |
| :--- | :--- | :--- | :--- | :--- |
| Zero-shot | XTTS-v2 | **0.3460** | 1.396 | 3.287 |
|  | CosyVoice2-0.5B | 0.4614 | 5.456 | 16.03 |
|  | FishAudio-S1-mini | 0.4037 | **1.257** | **2.364** |
|  | Qwen3-TTS-12Hz-0.6B-Base | 0.7497 | 2.076 | 4.037 |
|  | Qwen3-TTS-12Hz-1.7B-Base | 0.5767 | 1.488 | 2.884 |
| Single speaker (JSUT) | Tacotron2 | - | 0.05767 | - |
|  | Transformer-TTS | - | 2.249 | - |
|  | FastSpeech2 | - | 0.05767 | - |

## Speaker similarity
|  | CER=0 | CER<=10 | CER<=30 | CER<=50 | CER<=100 | no filterd |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| XTTS-v2 | 0.6267 | 0.6273 | 0.6218 | 0.6178 | 0.6155 | 0.6145 |
| CosyVoice2-0.5B | 0.7325 | 0.7251 | 0.7152 | 0.7087 | 0.6858 | 0.6848 |
| FishAudio-S1-mini | 0.6864 | 0.6833 | 0.6722 | 0.6646 | 0.6531 | 0.6440 |
| Qwen3-TTS-12Hz-0.6B-Base | 0.7419 | 0.7496 | 0.7451 | 0.7418 | 0.7354 | 0.7298 |
| Qwen3-TTS-12Hz-1.7B-Base | 0.7623 | 0.7614 | 0.7549 | 0.7539 | 0.7537 | **0.7530** |


# References
- [1] [Seed-TTS: A Family of High-Quality Versatile Speech Generation Models](https://arxiv.org/abs/2406.02430)
- [2] [UTMOS: UTokyo-SaruLab System for VoiceMOS Challenge 2022](https://arxiv.org/abs/2204.02152)
- [3] [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)
- [4] [WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900)
- [5] [ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification](https://arxiv.org/abs/2005.07143)
- [6] [JSUT corpus: free large-scale Japanese speech corpus for end-to-end speech synthesis](https://arxiv.org/abs/1711.00354)


# Acknowledgements
We would like to express our deepest gratitude to the authors of [**Seed-TTS-Eval**](https://github.com/BytedanceSpeech/seed-tts-eval). <br>
Their work served as a significant inspiration for the design and methodology of this benchmark.


# Citation
If you use this benchmark in your research, please cite our paper:

**Text:**
```text
Shuhei Imai, et al. "A Study on a High-Diﬃculty Japanese Text-to-Speech Corpus Specialized for Sequence Consistency Evaluation." Proceedings of the 2025 Autumn Meeting of the Acoustical Society of Japan, Sept. 2025.
```

**BibTeX (en):**
```bibtex
@inproceedings{imai2025jhard,
  author    = {Imai, Shuhei and Enomoto, Haruhisa and Kaneko, Takeshi and Nakamura, Taiki},
  title     = {A Study on a High-Diﬃculty Japanese Text-to-Speech Corpus Specialized for Sequence Consistency Evaluation},
  booktitle = {Proc. ASJ},
  year      = {2025},
  month     = {9}
}
```

**BibTeX (ja):**
```bibtex
@inproceedings{imai2025jhard,
  author    = {今井, 柊平 and 榎本, 悠久 and 金子, 剛士 and 中村, 泰貴},
  title     = {系列整合性評価に特化した高難度日本語テキスト音声合成コーパスの検討},
  booktitle = {音講論},
  year      = {2025},
  month     = {9}
}
```