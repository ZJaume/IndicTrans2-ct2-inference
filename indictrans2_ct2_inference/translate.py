from .engine import Model
from .flores_codes_map_indic import flores_codes
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError
from argparse import ArgumentParser
from pathlib import Path
from timeit import default_timer
import urllib.request
import logging
import tarfile
import sys
import os


logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def open_file(filepath, mode):
    if filepath.suffix == ".zst":
        import zstandard
        return zstandard.open(filepath, mode=mode)
    else:
        return open(filepath, mode=mode)


def download_and_extract(src_lang):
    filename_prefix = "{src}-{trg}-preprint.tar.gz"
    dirname = 'additional'
    if src_lang == "en" or src_lang == "eng_Latn":
        filename = filename_prefix.format(src="en", trg="indic")
    else:
        filename = filename_prefix.format(src="indic",trg="en")

    try:
        full_path = hf_hub_download(
            'ai4bharat/BPCC',
            filename,
            subfolder=dirname,
            revision='5bde309',
            repo_type="dataset",
        )
    except RepositoryNotFoundError as e:
        raise RuntimeError("Repository is gated, please make sure you are authenticated") from e

    full_path_extract_dir = full_path.removesuffix('.tar.gz')
    full_path = Path(full_path)
    model_type_folder = 'ct2_fp16_model'
    model_path = Path(full_path_extract_dir) / Path(model_type_folder)
    if not model_path.exists():
        logger.info("Extracting model from tarfile")
        with tarfile.open(full_path, 'r:gz') as tar:
            for member in tar.getmembers():
                if member.name.find(model_type_folder) >= 0:
                    logging.info(f"Extracting {member.name}")
                    tar.extract(member, path=full_path.parent, filter='data')

    return str(model_path)


class Translator():
    def __init__(
        self,
        src_lang,
        trg_lang,
        device: str = "cuda",
        device_index: list = [0],
        mini_batch_size: int = 4000,
        beam_size: int = 4,
    ):
        flores_reverse = {v:k for k,v in flores_codes.items()}
        flores_reverse["bn"] = "ben_Beng"
        flores_reverse["en"] = "eng_Latn"
        flores_reverse["hi"] = "hin_Deva"
        flores_reverse["or"] = "ory_Orya"
        flores_reverse["ur"] = "urd_Arab"
        if len(src_lang) < 3:
            src_lang = flores_reverse[src_lang]
        if len(trg_lang) < 3:
            trg_lang = flores_reverse[trg_lang]

        self.src_lang, self.trg_lang = src_lang, trg_lang
        model_path = download_and_extract(src_lang)
        self.model = Model(
            model_path,
            model_type="ctranslate2",
            device=device,
            device_index=device_index,
            mini_batch_size=mini_batch_size,
            beam_size=beam_size,
        )

    def batch_translate(self, batch, num_hypotheses = 1, src_lang = None, trg_lang = None):
        return self.model.batch_translate(
            batch,
            src_lang if src_lang else self.src_lang,
            trg_lang if trg_lang else self.trg_lang,
            num_hypotheses=num_hypotheses,
        )

def process_args():
    parser = ArgumentParser()
    parser.add_argument("src_lang")
    parser.add_argument("trg_lang")
    parser.add_argument("--gpus", type=str)
    parser.add_argument("--nbest", action="store_true", help="Produce fake nbest format")
    parser.add_argument("-i", "--input", type=Path, required=False, help="Instead of stdin, read from this file")
    parser.add_argument("-o", "--output", type=Path, required=False, help="Instead of stdout, wirte to this file")
    parser.add_argument("-M", "--maxi_batch", type=int, default=20000, required=False)
    parser.add_argument("-m", "--mini_batch", type=int, default=8000, required=False)
    parser.add_argument("-b", "--beam_size", type=int, default=4, required=False)
    args = parser.parse_args()
    args.gpus = list(map(int, args.gpus.split()))

    if args.input:
        args.input = open_file(args.input, mode='rt')
    else:
        args.input = sys.stdin
    if args.output:
        args.output = open_file(args.output, mode='wt')
    else:
        args.output = sys.stdout

    return args


def main():
    args = process_args()
    logger.info("Started")

    model = Translator(
        src_lang=args.src_lang,
        trg_lang=args.trg_lang,
        device_index=args.gpus,
        mini_batch_size=args.mini_batch,
        beam_size=args.beam_size
    )
    logging.info(f"Requested languages: {model.src_lang} {model.trg_lang}")

    def batched(stream):
        batch = []
        for line in stream:
            batch.append(line.strip())

            if len(batch) > args.maxi_batch:
                yield batch
                batch = []
        if batch:
            yield batch

    start_time = default_timer()
    total_toks = 0
    total_sents = 0
    total_bytes = 0
    line_number = 0
    for batch in batched(args.input):
        translated_batch = model.batch_translate(batch, args.src_lang, args.trg_lang)
        assert len(translated_batch) == len(batch), f"{len(translated_batch)},{len(batch)}"

        for line in translated_batch:
            if args.nbest:
                print(f"{line_number} ||| {line.strip()}", file=args.output)
            else:
                print(line.strip(), file=args.output)
            line_number += 1

        total_sents += len(batch)
        for i in batch:
            total_bytes += len(i.encode())

    elapsed = default_timer() - start_time
    toks_s = total_toks / elapsed
    sent_s = total_sents / elapsed
    bytes_s = total_bytes / elapsed
    logger.info(f"Total read: {total_toks} tokens - {total_sents} lines")
    logger.info(f"Throughput: {toks_s:.1f} tok/s - {sent_s:.1f} line/s")
    logger.info(f"Throughput: {bytes_s:.1f} bytes/s")
    logger.info(f"Elapsed time: {elapsed:.1f}")


if __name__ == "__main__":
    main()
