import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))

from pipeline.loader import load_data, get_X_y_from_df
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

train_df, dev_df, heldout_df = load_data()
X_train, y_train = get_X_y_from_df(train_df)
X_heldout, y_heldout = get_X_y_from_df(heldout_df)


def clean_tweet(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'https?\S+|www\.\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def make_text(series, mode):
    if mode == 'text':
        return series
    if mode == 'text+keyword':
        return series + ' ' + train_df['keyword'].fillna('').astype(str)
    if mode == 'text+location':
        return series + ' ' + train_df['location'].fillna('').astype(str)
    raise ValueError(mode)

texts = {
    'text': X_train,
    'text+keyword': X_train.astype(str) + ' ' + train_df['keyword'].fillna('').astype(str),
    'text+location': X_train.astype(str) + ' ' + train_df['location'].fillna('').astype(str),
}
heldout_texts = {
    'text': X_heldout,
    'text+keyword': X_heldout.astype(str) + ' ' + heldout_df['keyword'].fillna('').astype(str),
    'text+location': X_heldout.astype(str) + ' ' + heldout_df['location'].fillna('').astype(str),
}

preprocessors = [None, clean_tweet]
stop_words_opts = [None, 'english']
strip_accents_opts = [None, 'unicode']
ngram_opts = [(1, 1), (1, 2)]
max_features_opts = [5000, 6000, 7000, 8000, 10000, None]
min_df_opts = [1, 2]
max_df_opts = [1.0, 0.9, 0.8]
use_idf_opts = [True, False]
smooth_opts = [True, False]
sublinear_opts = [False, True]

target = 0.7574221578566256
best = None
count = 0
for mode in texts:
    for preprocessor in preprocessors:
        for stop_words in stop_words_opts:
            for strip_accents in strip_accents_opts:
                for ngram in ngram_opts:
                    for max_features in max_features_opts:
                        for min_df in min_df_opts:
                            for max_df in max_df_opts:
                                for use_idf in use_idf_opts:
                                    for smooth_idf in smooth_opts:
                                        for sublinear_tf in sublinear_opts:
                                            if not use_idf and smooth_idf:
                                                continue
                                            try:
                                                vec = TfidfVectorizer(
                                                    max_features=max_features,
                                                    min_df=min_df,
                                                    max_df=max_df,
                                                    ngram_range=ngram,
                                                    lowercase=True,
                                                    stop_words=stop_words,
                                                    strip_accents=strip_accents,
                                                    preprocessor=preprocessor,
                                                    use_idf=use_idf,
                                                    smooth_idf=smooth_idf,
                                                    sublinear_tf=sublinear_tf,
                                                )
                                                Xtr = vec.fit_transform(texts[mode])
                                                clf = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42, class_weight=None).fit(Xtr, y_train)
                                                pred = clf.predict(vec.transform(heldout_texts[mode]))
                                                f1 = f1_score(y_heldout, pred, pos_label=1)
                                            except Exception:
                                                continue
                                            count += 1
                                            if best is None or abs(f1-target) < abs(best[0]-target):
                                                best = (f1, {
                                                    'mode': mode,
                                                    'preprocessor': preprocessor,
                                                    'stop_words': stop_words,
                                                    'strip_accents': strip_accents,
                                                    'ngram': ngram,
                                                    'max_features': max_features,
                                                    'min_df': min_df,
                                                    'max_df': max_df,
                                                    'use_idf': use_idf,
                                                    'smooth_idf': smooth_idf,
                                                    'sublinear_tf': sublinear_tf,
                                                })
                                            if abs(f1-target) < 0.0005:
                                                print('FOUND', f1, best[1])
                                                raise SystemExit
print('count', count)
print('best', best[0], best[1])
