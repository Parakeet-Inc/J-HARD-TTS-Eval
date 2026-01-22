import json
import re
import unicodedata
from pathlib import Path

import jiwer
import pyopenjtalk
from omegaconf import OmegaConf

CONFIG_PATH = Path("./config.yaml")


def normalize_text(text: str) -> str:
    # Unify full-width and half-width characters
    text = unicodedata.normalize("NFKC", text)

    # If there are uppercase letters, convert them to lowercase
    text = text.lower()

    # Replace numbers 1-9 with Katakana
    # Assuming that numbers 10 and above are not used for evaluation
    text = (
        text.replace("1", "イチ")
        .replace("2", "ニ")
        .replace("3", "サン")
        .replace("4", "ヨン")
        .replace("5", "ゴ")
        .replace("6", "ロク")
        .replace("7", "ナナ")
        .replace("8", "ハチ")
        .replace("9", "キュウ")
    )

    # Roman letters are converted to Katakana
    text = (
        text.replace("a", "エー")
        .replace("b", "ビー")
        .replace("c", "シー")
        .replace("d", "ディー")
        .replace("e", "イー")
        .replace("f", "エフ")
        .replace("g", "ジー")
        .replace("h", "エイチ")
        .replace("i", "アイ")
        .replace("j", "ジェー")
        .replace("k", "ケー")
        .replace("l", "エル")
        .replace("m", "エム")
        .replace("n", "エヌ")
        .replace("o", "オー")
        .replace("p", "ピー")
        .replace("q", "キュー")
        .replace("r", "アール")
        .replace("s", "エス")
        .replace("t", "ティー")
        .replace("u", "ユー")
        .replace("v", "ブイ")
        .replace("w", "ダブリュー")
        .replace("x", "エックス")
        .replace("y", "ワイ")
        .replace("z", "ゼット")
    )

    # Convert to Katakana using pyopenjtalk to unify variations in Kanji, Hiragana, Katakana, and numbers
    # However, shorten texts longer than 200 characters
    if len(text) >= 200:
        text = text[:200]
    text = pyopenjtalk.g2p(text, kana=True)

    # Remove special characters
    code_regex = re.compile(
        "[!\"#$%&'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％]"
    )
    text = code_regex.sub("", text)

    # remove whitespaces
    text = text.replace("　", "")
    text = text.replace(" ", "")

    return text


def main():
    # load config
    cfg = OmegaConf.load(CONFIG_PATH)

    # cer for each subset
    for subset_name in cfg.subsets:
        # load ASR result
        with open(Path(cfg.result_dir_path_root) / subset_name / "asr.jsonl", "r") as f:
            jsonl_data = [json.loads(line) for line in f.read().splitlines()]

        text_list = []
        text_list_duplicated = []
        hypo_list = []
        hypo_cer_best_list = []
        hypo_cer_worst_list = []

        # main loop
        for data in jsonl_data:
            # normalize reference text
            text = data["text"]
            text_normalized = normalize_text(text)
            assert len(data["hypo_whisper"]) == len(data["hypo_reazon"])

            # initialize best and worst CER for the current sample
            cer_best = float("inf")
            cer_worst = float("-inf")
            hypo_cer_best = None
            hypo_cer_worst = None

            # calculate CER for each sample
            for hypo_whisper, hypo_reazon in zip(
                data["hypo_whisper"], data["hypo_reazon"]
            ):
                # normalize hypothesis
                hypo_whisper_normalized = normalize_text(hypo_whisper)
                hypo_reazon_normalized = normalize_text(hypo_reazon)

                # CER calculation
                cer_whisper = jiwer.cer(text_normalized, hypo_whisper_normalized) * 100
                cer_reazon = jiwer.cer(text_normalized, hypo_reazon_normalized) * 100

                # select better recognition result
                if cer_whisper < cer_reazon:
                    cer = cer_whisper
                    hypo = hypo_whisper_normalized
                else:
                    cer = cer_reazon
                    hypo = hypo_reazon_normalized

                # update best and worst CER and hypothesis
                if cer < cer_best:
                    cer_best = cer
                    hypo_cer_best = hypo
                if cer > cer_worst:
                    cer_worst = cer
                    hypo_cer_worst = hypo

                # append text and hypothesis
                text_list_duplicated.append(text_normalized)
                hypo_list.append(hypo)

            # append text, and best and worst CER's hypothesis
            text_list.append(text_normalized)
            hypo_cer_best_list.append(hypo_cer_best)
            hypo_cer_worst_list.append(hypo_cer_worst)

        # calculate micro CER for the subset
        cer_best = jiwer.cer(text_list, hypo_cer_best_list) * 100
        cer_average = jiwer.cer(text_list_duplicated, hypo_list) * 100
        cer_worst = jiwer.cer(text_list, hypo_cer_worst_list) * 100

        # output results
        with open(Path(cfg.result_dir_path_root) / subset_name / "cer.txt", "w") as f:
            f.write(
                f"CER_best: {cer_best:#.4g}[%]\n"
                f"CER_average: {cer_average:#.4g}[%]\n"
                f"CER_worst: {cer_worst:#.4g}[%]"
            )


if __name__ == "__main__":
    main()
