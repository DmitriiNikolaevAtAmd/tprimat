#!/usr/bin/env python3

import argparse
import json
import math
import os
import sys
import glob
import gzip
import multiprocessing
import time
from pathlib import Path

try:
    import nltk
    from nltk.tokenize.punkt import PunktLanguageVars
    nltk_available = True
except ImportError:
    PunktLanguageVars = object
    nltk_available = False

from transformers import AutoTokenizer


class CustomLanguageVars(PunktLanguageVars):
    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                          # any whitespace
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)        # or next token
        ))"""


class IdentitySplitter:
    def tokenize(self, *text):
        return text


class Encoder:
    
    tokenizer = None
    splitter = None
    
    def __init__(self, args):
        self.args = args
    
    def initializer(self):
        Encoder.tokenizer = AutoTokenizer.from_pretrained(
            self.args.tokenizer_model,
            token=os.environ.get("HF_TOKEN"),
            trust_remote_code=True
        )
        
        if self.args.split_sentences:
            if not nltk_available:
                print("NLTK is not available for sentence splitting.")
                sys.exit(1)
            
            if os.environ.get("NLTK_DATA"):
                library = os.path.join(
                    os.environ.get("NLTK_DATA"),
                    "tokenizers", "punkt", f"{self.args.lang}.pickle"
                )
                url = f"file:{library}"
            else:
                library = os.path.join("tokenizers", "punkt", f"{self.args.lang}.pickle")
                url = f"nltk:{library}"
            
            splitter = nltk.load(url)
            if self.args.keep_newlines:
                Encoder.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                    train_text=splitter._params,
                    lang_vars=CustomLanguageVars()
                )
            else:
                Encoder.splitter = splitter
        else:
            Encoder.splitter = IdentitySplitter()
    
    def split(self, json_line):
        data = json.loads(json_line)
        output = {}
        for key in self.args.json_keys:
            text = data[key]
            max_len = 1000000
            tokens_list = [
                Encoder.splitter.tokenize(text[i:i + max_len])
                for i in range(0, len(text), max_len)
            ]
            output[key] = [tokens for partial in tokens_list for tokens in partial]
        return json.dumps(output), len(json_line)
    
    def encode(self, json_line):
        data = json.loads(json_line)
        ids = {}
        lens = {}
        
        for key in self.args.json_keys:
            text = data[key]
            sentences = text if isinstance(text, list) else [text]
            
            doc_ids = []
            sentence_lens = []
            
            for sentence in sentences:
                sentence_ids = Encoder.tokenizer.encode(sentence, add_special_tokens=False)
                if len(sentence_ids) > 0:
                    doc_ids.extend(sentence_ids)
                    sentence_lens.append(len(sentence_ids))
            
            if len(doc_ids) > 0 and self.args.append_eod:
                eod_id = Encoder.tokenizer.eos_token_id or 0
                doc_ids.append(eod_id)
                sentence_lens[-1] += 1
            
            ids[key] = doc_ids
            lens[key] = sentence_lens
        
        return ids, lens, len(json_line)


class IndexedDatasetBuilder:
    
    def __init__(self, output_prefix: str):
        self.output_prefix = output_prefix
        self.data = []
        self.sizes = []
        self.doc_indices = [0]
    
    def add_document(self, token_ids: list, sentence_lens: list = None):
        import numpy as np
        self.data.extend(token_ids)
        self.sizes.append(len(token_ids))
        self.doc_indices.append(self.doc_indices[-1] + len(token_ids))
    
    def finalize(self):
        import numpy as np
        
        # Write binary data
        bin_file = f"{self.output_prefix}.bin"
        idx_file = f"{self.output_prefix}.idx"
        
        data_array = np.array(self.data, dtype=np.int32)
        data_array.tofile(bin_file)
        
        sizes_array = np.array(self.sizes, dtype=np.int32)
        doc_idx_array = np.array(self.doc_indices, dtype=np.int64)
        
        with open(idx_file, 'wb') as f:
            f.write(b'MMIDIDX\x00')  # Magic + version
            f.write(np.array([1], dtype=np.int64).tobytes())  # Version
            f.write(np.array([4], dtype=np.int8).tobytes())  # dtype code (int32)
            f.write(np.array([len(self.sizes)], dtype=np.int64).tobytes())  # num docs
            f.write(np.array([len(self.doc_indices)], dtype=np.int64).tobytes())  # num elements
            f.write(sizes_array.tobytes())
            f.write(doc_idx_array.tobytes())
        
        print(f"[Info] Wrote {len(self.sizes)} documents, {len(self.data)} tokens")
        print(f"       Binary: {bin_file}")
        print(f"       Index:  {idx_file}")


def process_file(args):
    """Process a single JSON/JSONL file."""
    print(f"[Info] Processing: {args.input}")
    
    encoder = Encoder(args)
    encoder.initializer()
    
    if args.input.endswith('.gz'):
        fin = gzip.open(args.input, 'rt', encoding='utf-8')
    else:
        fin = open(args.input, 'r', encoding='utf-8')
    
    builders = {}
    for key in args.json_keys:
        level = "sentence" if args.split_sentences else "document"
        output_prefix = f"{args.output_prefix}_{key}_{level}"
        builders[key] = IndexedDatasetBuilder(output_prefix)
    
    start_time = time.time()
    total_bytes = 0
    doc_count = 0
    
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
    
    for doc_ids, sentence_lens, bytes_processed in pool.imap(encoder.encode, fin, 32):
        total_bytes += bytes_processed
        doc_count += 1
        
        for key in doc_ids.keys():
            builders[key].add_document(doc_ids[key], sentence_lens.get(key))
        
        if doc_count % args.log_interval == 0:
            elapsed = time.time() - start_time
            mbs = total_bytes / elapsed / 1024 / 1024
            print(f"[Info] Processed {doc_count} documents ({mbs:.2f} MB/s)")
    
    fin.close()
    pool.close()
    pool.join()
    
    for key, builder in builders.items():
        builder.finalize()
    
    elapsed = time.time() - start_time
    print(f"[Info] Completed in {elapsed:.1f}s: {doc_count} documents")


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize dataset for Megatron-LM training"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input JSON/JSONL file"
    )
    parser.add_argument(
        "--output-prefix", "-o",
        type=str,
        required=True,
        help="Output prefix for .bin/.idx files"
    )
    parser.add_argument(
        "--tokenizer-model",
        type=str,
        required=True,
        help="HuggingFace tokenizer model (e.g., meta-llama/Meta-Llama-3-8B)"
    )
    parser.add_argument(
        "--json-keys",
        nargs="+",
        default=["text"],
        help="JSON keys to extract (default: text)"
    )
    parser.add_argument(
        "--split-sentences",
        action="store_true",
        help="Split documents into sentences"
    )
    parser.add_argument(
        "--keep-newlines",
        action="store_true",
        help="Keep newlines when splitting sentences"
    )
    parser.add_argument(
        "--append-eod",
        action="store_true",
        help="Append EOD token to each document"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="english",
        help="Language for NLTK sentence splitting (default: english)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, os.cpu_count() - 1),
        help="Number of worker processes"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1000,
        help="Logging interval (default: 1000)"
    )
    args = parser.parse_args()
    
    if args.split_sentences and nltk_available:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
    
    output_dir = Path(args.output_prefix).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    process_file(args)


if __name__ == "__main__":
    main()
