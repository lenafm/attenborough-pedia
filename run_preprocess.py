# import modules
import os
import argparse
import pysrt
import spacy
from copy import deepcopy

from utils.preprocess_text import preprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str,
                        help='input directory', default='data/srt')
    parser.add_argument('--outdir', type=str,
                        help='output directory', default='data/preprocessed_subtitles')
    args = parser.parse_args()

    indir = args.indir
    outdir = args.outdir
    nlp = spacy.load("en_core_web_lg")

    filenames = os.listdir(indir)
    for file in filenames:
        print('Processing file {}...'.format(file))
        subtitles = pysrt.open(os.path.join(indir, file), encoding='latin-1')
        utterances = [subtitle.text for subtitle in subtitles]
        corpus = nlp.pipe(utterances, batch_size=26000)
        processed = preprocess(corpus)
        preprocessed_subtitles = deepcopy(subtitles)

        for subtitle, proc_subtitle in zip(preprocessed_subtitles, processed):
            subtitle.text = " ".join(proc_subtitle)

        preprocessed_subtitles.save(os.path.join(outdir, '{}.txt'.format(file[:-4])))
