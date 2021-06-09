import pandas as pd
import numpy as np
from gensim.models.phrases import Phrases, Phraser
import multiprocessing

from gensim.models import Word2Vec

df = pd.read_csv("clean_gensim.csv")
sent = [row.split() for row in df['clean_text']]
sent
