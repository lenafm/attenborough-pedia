import pysrt
import spacy
from tqdm.auto import tqdm
from copy import deepcopy
from tabulate import tabulate

from utils.preprocess_text import preprocess, token_filter

subtitles = pysrt.open("./Copy of 08. Lords of the air.srt")
nlp = spacy.load("en_core_web_lg")

utterances = [subtitle.text for subtitle in subtitles]

corpus = nlp.pipe(utterances, batch_size=26000)

processed = preprocess(corpus)

preprocessed_subtitles = deepcopy(subtitles)

for subtitle, proc_subtitle in zip(preprocessed_subtitles, processed):
    subtitle.text = " ".join(proc_subtitle)

preprocessed_subtitles.save('./preprocessed_sub.txt')
