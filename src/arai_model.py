# modified from https://github.com/utanaka2000/fairseq/blob/japanese_bart_pretrained_model/fairseq_cli/interactive.py

import os
import math
import torch
import zenhan
import numpy as np
import sentencepiece
from pyknp import Juman
from pathlib import Path
from collections import namedtuple, Counter


from fairseq.data import encoders
from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.token_generation_constraints import pack_constraints, unpack_constraints


Batch = namedtuple('Batch', 'ids src_tokens src_lengths constraints')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')

parser = options.get_interactive_generation_parser("translation_from_pretrained_bart")
args = options.parse_args_and_arch(parser)
model_path = Path(__file__).parent.parent / 'model/arai/'
args.data = str(model_path)
args.path = str(model_path / "checkpoint_best.pt")
args.task = str("translation_from_pretrained_bart")

args.bpe_model = str(model_path / "sp.model")
args.bpe_dict = str(model_path / "dict.txt")

def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, 'symbols_to_strip_from_output'):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}


def make_batches(lines, args, task, max_positions, encode_fn):
    def encode_fn_target(x):
        return encode_fn(x)

    if args.constraints:
        # Strip (tab-delimited) contraints, if present, from input lines,
        # store them in batch_constraints
        batch_constraints = [list() for _ in lines]
        for i, line in enumerate(lines):
            if "\t" in line:
                lines[i], *batch_constraints[i] = line.split("\t")

        # Convert each List[str] to List[Tensor]
        for i, constraint_list in enumerate(batch_constraints):
            batch_constraints[i] = [task.target_dictionary.encode_line(
                encode_fn_target(constraint),
                append_eos=False,
                add_if_not_exist=False,
            ) for constraint in constraint_list]

    tokens = [
        task.source_dictionary.encode_line(
            encode_fn(src_str), add_if_not_exist=False
        ).long()
        for src_str in lines
    ]

    if args.constraints:
        constraints_tensor = pack_constraints(batch_constraints)
    else:
        constraints_tensor = None

    lengths = [t.numel() for t in tokens]
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(tokens, lengths, constraints=constraints_tensor),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        ids = batch['id']
        src_tokens = batch['net_input']['src_tokens']
        src_lengths = batch['net_input']['src_lengths']
        constraints = batch.get("constraints", None)

        yield Batch(
            ids=ids,
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            constraints=constraints,
        )


utils.import_user_module(args)

if args.buffer_size < 1:
    args.buffer_size = 1
if args.max_tokens is None and args.max_sentences is None:
    args.max_sentences = 1

assert not args.sampling or args.nbest == args.beam, \
    '--sampling requires --nbest to be equal to --beam'
assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
    '--max-sentences/--batch-size cannot be larger than --buffer-size'

# Fix seed for stochastic decoding
if args.seed is not None and not args.no_seed_provided:
    np.random.seed(args.seed)
    utils.set_torch_seed(args.seed)

#use_cuda = torch.cuda.is_available() and not args.cpu
use_cuda = False

# Setup task, e.g., translation
task = tasks.setup_task(args)

# Load ensemble
models, _model_args = checkpoint_utils.load_model_ensemble(
    args.path.split(os.pathsep),
    arg_overrides=eval(args.model_overrides),
    task=task,
    suffix=getattr(args, "checkpoint_suffix", ""),
)

# Set dictionaries
src_dict = task.source_dictionary
tgt_dict = task.target_dictionary

# Optimize ensemble for generation
for model in models:
    model.prepare_for_inference_(args)
    if args.fp16:
        model.half()
    if use_cuda:
        model.cuda()

# Initialize generator
generator = task.build_generator(models, args)

# Handle tokenization and BPE
#tokenizer = encoders.build_tokenizer(args)
#bpe = encoders.build_bpe(args)

tokenizer = Juman()
bpe = sentencepiece.SentencePieceProcessor()
bpe.Load(args.bpe_model)
vocabs=[]
with open(args.bpe_dict) as f:
    for line in f:
        vocabs.append(line.strip().split()[0])
bpe.set_vocabulary(vocabs)

def juman_split(line, jumanpp):
   result = jumanpp.analysis(line)
   return ' '.join([mrph.midasi for mrph in result.mrph_list()])

def bpe_encode(line, spm):
    return ' '.join(spm.EncodeAsPieces(line.strip()))

def encode_fn(x):
    if tokenizer is not None:
        x = juman_split(x, tokenizer)
    if bpe is not None:
        x = bpe_encode(x, bpe)
    return x

def decode_fn(x):
    if bpe is not None:
        x = bpe.decode(x.split(" "))
    x = x.replace(" ", "")
    return x

# Load alignment dictionary for unknown word replacement
# (None if no unknown word replacement, empty if no path to align dictionary)
align_dict = utils.load_align_dict(args.replace_unk)

max_positions = utils.resolve_max_positions(
    task.max_positions(),
    *[model.max_positions() for model in models]
)


def make_arai_reply(input_texts):
    results_detok = []
    start_id = 0
    for inputs in input_texts:
        results = []
        result_detok = Counter()
        for batch in make_batches(inputs, args, task, max_positions, encode_fn):
            bsz = batch.src_tokens.size(0)
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            constraints = batch.constraints
            if use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()
                if constraints is not None:
                    constraints = constraints.cuda()

            sample = {
                'net_input': {
                    'src_tokens': src_tokens,
                    'src_lengths': src_lengths,
                },
            }

            translations = task.inference_step(generator, models, sample, constraints=constraints)
            list_constraints = [[] for _ in range(bsz)]
            if args.constraints:
                list_constraints = [unpack_constraints(c) for c in constraints]
            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
                constraints = list_constraints[i]
                results.append((start_id + id, src_tokens_i, hypos, {"constraints": constraints}))

        # sort output to match input order
        for id_, src_tokens, hypos, info in sorted(results, key=lambda x: x[0]):
            if src_dict is not None:
                src_str = src_dict.string(src_tokens, args.remove_bpe)

            # Process top predictions
            for hypo in hypos[:min(len(hypos), args.nbest)]:
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'],
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=args.remove_bpe,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                )
                detok_hypo_str = decode_fn(hypo_str)
                score = hypo['score'] / math.log(2)  # convert to base 2
                
                result_detok[detok_hypo_str] = score
                #result_detok.append((detok_hypo_str, score))
        results_detok.append(result_detok)
        start_id += len(inputs)

    

    return [{"reply_text": results_detok[0].most_common(1)[0][0]}]



if __name__ == '__main__':
    reply = make_arai_reply([["今日も良い天気ですね\n今日も元気に頑張りましょう"]])#[0].most_common(1)[0][0]
    print(reply)