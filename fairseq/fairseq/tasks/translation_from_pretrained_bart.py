# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import itertools
import os
import logging

from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    data_utils,
    encoders,
    indexed_dataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
)

from .translation import TranslationTask
from . import register_task


logger = logging.getLogger(__name__)


@register_task('translation_from_pretrained_bart')
class TranslationFromPretrainedBARTTask(TranslationTask):
    """
    Translate from source language to target language with a model initialized with a monolingual or multilingual pretrain.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationTask.add_args(parser)
        parser.add_argument('--langs', default=None,  metavar='LANG',
                            help='comma-separated list of monolingual language, '
                                 'for example, "en,de,fr". These should match the '
                                 'langs from pretraining (and be in the same order). '
                                 'You should always add all pretraining language idx '
                                 'during finetuning. This is required only when using '
                                 'a multilingual pretrained model.')
        parser.add_argument('--prepend-bos', action='store_true',
                            help='prepend bos token to each sentence, which matches '
                                 'mBART pretraining')
        parser.add_argument('--src-suffix', type=str, default=None,
                            help='specify file suffix. This is for using a multilingual'
                                 'model when the source and targe language are the same')
        parser.add_argument('--tgt-suffix', type=str, default=None,
                            help='specify file suffix. This is for using a multilingual'
                                 'model when the source and targe language are the same')
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        for d in [src_dict, tgt_dict]:
            d.add_symbol('<mask>')

        if self.args.langs is not None:
            self.langs = args.langs.split(',')
            for d in [src_dict, tgt_dict]:
                for l in self.langs:
                    d.add_symbol('[{}]'.format(l))

    def load_langpair_dataset(
        self,
        data_path, split,
        src, src_dict,
        tgt, tgt_dict,
        combine, dataset_impl, upsample_primary,
        left_pad_source, left_pad_target, max_source_positions,
        max_target_positions, prepend_bos=False, load_alignments=False,
        truncate_source=False, append_source_id=False,
        num_buckets=0,
        shuffle=True,
        pad_to_multiple=1,
    ):

        def split_exists(split, src, tgt, lang, data_path):
            filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
            return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

        src_datasets = []
        tgt_datasets = []

        for k in itertools.count():
            split_k = split + (str(k) if k > 0 else '')

            # infer langcode
            if split_exists(split_k, src, tgt, src, data_path):
                prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
            elif split_exists(split_k, tgt, src, src, data_path):
                prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
            else:
                if k > 0:
                    break
                else:
                    raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

            src_dataset = data_utils.load_indexed_dataset(prefix + src, src_dict, dataset_impl)
            if truncate_source:
                src_dataset = AppendTokenDataset(
                    TruncateDataset(
                        StripTokenDataset(src_dataset, src_dict.eos()),
                        max_source_positions - 1,
                    ),
                    src_dict.eos(),
                )
            src_datasets.append(src_dataset)

            tgt_dataset = data_utils.load_indexed_dataset(prefix + tgt, tgt_dict, dataset_impl)
            if tgt_dataset is not None:
                tgt_datasets.append(tgt_dataset)

            logger.info('{} {} {}-{} {} examples'.format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            ))

            if not combine:
                break

        assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

        if len(src_datasets) == 1:
            src_dataset = src_datasets[0]
            tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
        else:
            sample_ratios = [1] * len(src_datasets)
            sample_ratios[0] = upsample_primary
            src_dataset = ConcatDataset(src_datasets, sample_ratios)
            if len(tgt_datasets) > 0:
                tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
            else:
                tgt_dataset = None

        if prepend_bos:
            assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
            src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
            if tgt_dataset is not None:
                tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

        eos = None
        if append_source_id:
            src_dataset = AppendTokenDataset(src_dataset, src_dict.index('[{}]'.format(self.args.source_lang)))
            if tgt_dataset is not None:
                tgt_dataset = AppendTokenDataset(tgt_dataset, tgt_dict.index('[{}]'.format(self.args.target_lang)))
            eos = tgt_dict.index('[{}]'.format(self.args.target_lang))

        align_dataset = None
        if load_alignments:
            align_path = os.path.join(data_path, '{}.align.{}-{}'.format(split, src, tgt))
            if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
                align_dataset = data_utils.load_indexed_dataset(align_path, None, dataset_impl)

        tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
        return LanguagePairDataset(
            src_dataset, src_dataset.sizes, src_dict,
            tgt_dataset, tgt_dataset_sizes, tgt_dict,
            left_pad_source=left_pad_source,
            left_pad_target=left_pad_target,
            align_dataset=align_dataset, eos=eos,
            num_buckets=num_buckets,
            shuffle=shuffle,
            pad_to_multiple=pad_to_multiple,
        )

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        if self.args.src_suffix is not None and self.args.tgt_suffix is not None:
            src, tgt = self.args.src_suffix, self.args.tgt_suffix
        else:
            src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = self.load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=getattr(self.args, 'max_source_positions', 1024),
            max_target_positions=getattr(self.args, 'max_target_positions', 1024),
            load_alignments=self.args.load_alignments,
            prepend_bos=self.args.prepend_bos, 
            append_source_id=self.args.langs is not None
            )
        
    def build_generator(self, models, args):
        if getattr(args, 'score_reference', False):
            from fairseq.sequence_scorer import SequenceScorer
            return SequenceScorer(
                self.target_dictionary,
                eos=self.tgt_dict.eos() if self.args.langs is None else self.tgt_dict.index('[{}]'.format(self.args.target_lang))
            )
        else:
            from fairseq.sequence_generator import SequenceGenerator
            return SequenceGenerator(
                models,
                self.target_dictionary,
                beam_size=getattr(args, 'beam', 5),
                max_len_a=getattr(args, 'max_len_a', 0),
                max_len_b=getattr(args, 'max_len_b', 200),
                min_len=getattr(args, 'min_len', 1),
                normalize_scores=(not getattr(args, 'unnormalized', False)),
                len_penalty=getattr(args, 'lenpen', 1),
                unk_penalty=getattr(args, 'unkpen', 0),
                temperature=getattr(args, 'temperature', 1.),
                match_source_len=getattr(args, 'match_source_len', False),
                no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
                eos=self.tgt_dict.eos() if self.args.langs is None else self.tgt_dict.index('[{}]'.format(self.args.target_lang))
            )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        if self.args.prepend_bos:
            bos_id = self.source_dictionary.bos()
            source_tokens = []
            for s_t in src_tokens:
                s_t = torch.cat([s_t.new(1).fill_(bos_id), s_t])
                source_tokens.append(s_t)
            src_tokens = source_tokens

        if self.args.langs is None:
            dataset = LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary,
                                          tgt_dict=self.target_dictionary,
                                          constraints=constraints)
        else:
            src_lang_id = self.source_dictionary.index('[{}]'.format(self.args.source_lang))
            source_tokens = []
            for s_t in src_tokens:
                s_t = torch.cat([s_t, s_t.new(1).fill_(src_lang_id)])
                source_tokens.append(s_t)
            dataset = LanguagePairDataset(source_tokens, src_lengths, self.source_dictionary,
                                          tgt_dict=self.target_dictionary,
                                          constraints=constraints)
        return dataset
