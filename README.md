# Contents
- Main data set: [changemyview-sociopol-processed.pickle.gz](https://github.com/corradomonti/10-dim-of-op-change/releases/tag/changemyview-sociopol-processed)  149 MB
- [Human-labelled set of sociopolitical subreddits](data/top-2000-sfw-subreddits-labelled.csv)
- [Sociopolitical classifier](data/vocabulary-classifier.pickle)
- [Code taken from the original paper repository](reproduce/from_original)
- [Code modified so that the results match the paper's results](reproduce/modified)

# Reproduce Results
1. Download [changemyview-sociopol-processed.pickle.gz](https://github.com/corradomonti/10-dim-of-op-change/releases/tag/changemyview-sociopol-processed), and place it in the data folder.
2. `pip install -r requirements.txt`
3. In order to reproduce *figure 1* run `reproduce/modified/reproduce_figure1.py` and observe the output directory inside `reproduce/modified`.
4. In order to reproduce *table 2* run `reproduce/modified/reproduce_table2.py` and observe the output directory inside `reproduce/modified`.
