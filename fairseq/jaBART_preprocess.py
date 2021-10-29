import sys
import argparse
import zenhan
from pyknp import Juman
import sentencepiece

def juman_split(line, jumanpp):
   result = jumanpp.analysis(line)
   return ' '.join([mrph.midasi for mrph in result.mrph_list()])

def bpe_encode(line, spm):
    return ' '.join(spm.EncodeAsPieces(line.strip()))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bpe_model', required=True)
    parser.add_argument('--bpe_dict', required=True)
    args = parser.parse_args()

    jumanpp = Juman()
    spm = sentencepiece.SentencePieceProcessor()
    spm.Load(args.bpe_model)
    vocabs=[]
    with open(args.bpe_dict) as f:
        for line in f:
            vocabs.append(line.strip().split()[0])
    spm.set_vocabulary(vocabs)

    for line in sys.stdin:
        line = line.strip()
        line = zenhan.h2z(line)
        line = juman_split(line, jumanpp)
        line = bpe_encode(line, spm)
        print(line)
        
if __name__ == '__main__':
    main()
