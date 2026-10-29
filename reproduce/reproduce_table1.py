%matplotlib inline

import random
import math
import time
import os
from os.path import join
import sys
import pandas as pd
pd.set_option('display.max_columns', 500)
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy
import bz2

from nltk.tokenize import sent_tokenize

from nltk.tokenize import TweetTokenizer
from nltk.tokenize import RegexpTokenizer
tweet_tokenize = TweetTokenizer().tokenize
regex_tokenize = RegexpTokenizer(r'\w+').tokenize

import matplotlib.colors as mcolors
import pickle

from tqdm.notebook import tqdm_notebook

#import statsmodels as sm
import statsmodels.api as sm
import scipy.stats as st
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS
from sklearn import linear_model
from sklearn import preprocessing

dimensions = ['social_support', 'knowledge', 'conflict', 'power', 'similarity', 
              'fun', 'respect', 'trust', 'identity', 'romance']
dimensions_redux = ['social_support', 'knowledge', 'conflict', 'power', 'similarity', 
              'fun', 'respect', 'trust', 'identity']
dimensions_names = ['support', 'knowledge', 'conflict', 'power', 'similarity', 
              'fun', 'status', 'trust', 'identity', 'romance']
dimensions_names_redux = ['support', 'knowledge', 'conflict', 'power', 'similarity', 
                          'fun', 'status', 'trust', 'identity']
dims2name = {'social_support':'support', 'romance':'romance', 'identity':'identity' , 'similarity':'similarity',
             'conflict':'conflict','fun':'fun','knowledge':'knowledge','respect':'status','trust':'trust','power':'power'}

additional_dims = ['vader_sentiment', 'number_of_words']
additional_dims_names = ['sentiment', 'num_words']
additional_dims_names_binarized = ['sentiment_pos', 'sentiment_neg'] 

def ci(x, confidence=0.95):
    avg = np.mean(x)
    low, high = st.t.interval(confidence, len(x)-1, loc=avg, scale=st.sem(x))
    return (avg, high-avg)
    
#dimension-dependent weights
def standardized_dimension_score(x,mu,std):
    zscore = (x-mu)/std
    if zscore >= 0:
        return 1/(1+zscore)
    else:
        zscore = -zscore
        return 2 - (1/(1+zscore))
D = pd.read_pickle('changemyview-sociopol-processed.pickle.gz')
df_comments = D[['post_id', 'comment_id', 'vader_sentiment_comment','number_of_words','score_comment','is_op','has_delta','has_answer_by_op','log_number_of_words','support_comment','knowledge_comment','conflict_comment','power_comment','similarity_comment','fun_comment','status_comment','trust_comment','identity_comment','romance_comment','sentiment_pos_comment','sentiment_neg_comment','support_weighted_comment','knowledge_weighted_comment','conflict_weighted_comment','power_weighted_comment','similarity_weighted_comment','fun_weighted_comment','status_weighted_comment','trust_weighted_comment','identity_weighted_comment','romance_weighted_comment','sentiment_pos_weighted_comment','sentiment_neg_weighted_comment']]
df_comments.columns = ['post_id', 'comment_id','vader_sentiment','number_of_words','score_comment','is_op','has_delta','has_answer_by_op','log_number_of_words','support','knowledge','conflict','power','similarity','fun','status','trust','identity','romance','sentiment_pos','sentiment_neg','support_weighted','knowledge_weighted','conflict_weighted','power_weighted','similarity_weighted','fun_weighted','status_weighted','trust_weighted','identity_weighted','romance_weighted','sentiment_pos_weighted','sentiment_neg_weighted']
df_comments['nodimensions'] = df_comments[[f'{d}' for d in dimensions_names_redux]].sum(axis=1)
df_comments['nodimensions'] = df_comments['nodimensions'].apply(lambda x : 0 if x>0 else 1)
log_lengths = df_comments[df_comments['nodimensions'] == 1]['log_number_of_words']
avg = np.mean(log_lengths)
stdev = np.std(log_lengths)
df_comments[f'nodimensions_weighted'] = df_comments['log_number_of_words'].apply(lambda x: standardized_dimension_score(x,avg,stdev)) * df_comments['nodimensions']
df_no_delta = df_comments[df_comments['has_delta'] == False].dropna()
df_delta = df_comments[df_comments['has_delta'] == True].dropna()
df_delta = df_delta[df_delta['is_op'] == False]
df_no_delta = df_no_delta[df_no_delta['is_op'] == False]
fig, ax = plt.subplots(1, 1, figsize=(7, 6))

y_oddsratio = []
y_ci_OR = []
y_ratio = []
y_delta = []
y_nodelta = []
y_ci_delta = []
y_ci_nodelta = []
colors = []

all_dims = dimensions_names_redux + additional_dims_names_binarized + ['nodimensions']

for dim in all_dims:
    scores_delta = df_delta[f'{dim}_weighted']
    scores_no_delta = df_no_delta[f'{dim}_weighted']

    avg_delta, ci_delta = ci(scores_delta)
    avg_no_delta, ci_no_delta = ci(scores_no_delta)

    OR = (avg_delta / (1 - avg_delta)) / (avg_no_delta / (1 - avg_no_delta))
    log_OR = np.log(OR)

    y_oddsratio.append(OR)
    y_ratio.append((avg_delta / avg_no_delta) - 1)
    y_delta.append(avg_delta / avg_no_delta)
    y_nodelta.append(avg_no_delta)
    y_ci_delta.append(ci_delta / avg_no_delta)
    y_ci_nodelta.append(ci_no_delta / avg_no_delta)

    _dim_delta = int(avg_delta * len(scores_delta))
    _no_dim_delta = len(scores_delta) - _dim_delta
    _dim_no_delta = int(avg_no_delta * len(scores_no_delta))
    _no_dim_no_delta = len(scores_no_delta) - _dim_no_delta

    standard_error_log_OR = np.sqrt(
        1 / _dim_delta
        + 1 / _no_dim_delta
        + 1 / _dim_no_delta
        + 1 / _no_dim_no_delta
    )

    ci_log_OR = 1.96 * standard_error_log_OR

    ci_OR_upper = np.exp(log_OR + ci_log_OR)
    ci_OR_lower = np.exp(log_OR - ci_log_OR)

    ci_OR = ci_OR_upper - OR
    y_ci_OR.append(ci_OR)

    if avg_delta > avg_no_delta:
        clr = 'b'
    else:
        clr = 'r'

    colors.append(clr)


L = list(zip(
    y_oddsratio,
    y_ci_OR,
    y_ratio,
    y_delta,
    y_ci_delta,
    y_nodelta,
    y_ci_nodelta,
    colors,
    all_dims
))

L.sort(key=lambda x: x[0], reverse=True)

labels_ordered = []

for i, l in enumerate(L):
    y_odds, y_odds_ci, y_rat, y_d, err_delta, y_nod, err_nodelta, color, label = l

    labels_ordered.append(label)

    ax.errorbar(
        y_odds,
        i,
        xerr=y_odds_ci,
        elinewidth=1,
        fmt='o',
        markersize=4,
        color=color,
        capsize=3,
        zorder=2
    )

    ax.annotate(
        f'{y_odds:.2f}',
        (y_odds, i),
        textcoords='offset points',
        xytext=(0, 7),
        ha='center',
        va='bottom',
        fontsize=10,
        color='black'
    )

ax.set_title('Odds ratio (comments)', size=18)
ax.set_yticks(range(len(labels_ordered)))
ax.set_yticklabels(labels_ordered)
ax.set_xlabel('(a)', size=18)

ax.set_xlim(0, 3.0)
ax.set_xticks(np.arange(0.0, 3.01, 0.5))
ax.xaxis.tick_bottom()
ax.xaxis.set_label_position('bottom')

ax.grid(axis='x', linestyle='-', linewidth=0.7, alpha=0.4, zorder=0)

ax.axvline(1, linestyle='--', linewidth=1, color='gray', zorder=1)

ax.invert_yaxis()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('odds_ratio_comments.png', dpi=300, bbox_inches='tight')
plt.show()
df_posts = D[['post_id', 'vader_sentiment_post','score_post','number_of_words_post','log_number_of_words_post','support_post','knowledge_post','conflict_post','power_post','similarity_post','fun_post','status_post','trust_post','identity_post','romance_post','sentiment_pos_post','sentiment_neg_post','support_weighted_post','knowledge_weighted_post','conflict_weighted_post','power_weighted_post','similarity_weighted_post','fun_weighted_post','status_weighted_post','trust_weighted_post','identity_weighted_post','romance_weighted_post','sentiment_pos_weighted_post','sentiment_neg_weighted_post','num_links','length','log_num_links','log_length','links_per_length']]
df_posts.columns = ['post_id', 'vader_sentiment','score_post','number_of_words_post','log_number_of_words_post','support','knowledge','conflict','power','similarity','fun','status','trust','identity','romance','sentiment_pos','sentiment_neg','support_weighted','knowledge_weighted','conflict_weighted','power_weighted','similarity_weighted','fun_weighted','status_weighted','trust_weighted','identity_weighted','romance_weighted','sentiment_pos_weighted','sentiment_neg_weighted','num_links','length','log_num_links','log_length','links_per_length']
df_posts = df_posts.drop_duplicates(subset=['post_id'])
df_comments_shuffled = df_comments.copy(deep=True)
df_comments_shuffled['post_id'] = np.random.permutation(df_comments_shuffled['post_id'].values)
df_paired_shuffled = pd.merge(df_comments_shuffled, df_posts, left_on='post_id', right_on='post_id', how='inner', suffixes=('_comment','_post'))
df_paired = D.copy()
df_paired = df_paired[(df_paired['has_delta'] == True) | (df_paired['has_answer_by_op'] == True)] 
df_paired_shuffled = df_paired_shuffled[(df_paired_shuffled['has_delta'] == True) | (df_paired_shuffled['has_answer_by_op'] == True)] 
for t in ['post', 'comment']:
    df_paired[f'nodimensions_{t}'] = sum([df_paired[f'{d}_{t}'] for d in dimensions_names_redux])
    df_paired[f'nodimensions_{t}'] = df_paired[f'nodimensions_{t}'].apply(lambda x : 0 if x>0 else 1)
    df_paired[f'nodimensions_weighted_{t}'] = df_paired[f'nodimensions_{t}']
    
    df_paired[f'anydimensions_{t}'] = sum([df_paired[f'{d}_{t}'] for d in dimensions_names_redux])
    df_paired[f'anydimensions_{t}'] = df_paired[f'anydimensions_{t}'].apply(lambda x : 1 if x>0 else 0)
    df_paired[f'anydimensions_weighted_{t}'] = df_paired[f'anydimensions_{t}']

    if t == 'post':
        log_lengths = df_paired[df_paired[f'nodimensions_{t}'] == 1]['log_number_of_words_post']
        avg = np.mean(log_lengths)
        stdev = np.std(log_lengths)
        df_paired[f'nodimensions_weighted_{t}'] = df_paired['log_number_of_words_post'].apply(lambda x: standardized_dimension_score(x,avg,stdev)) * df_paired[f'nodimensions_{t}']
        df_paired[f'anydimensions_weighted_{t}'] = np.sum([df_paired[f'{d}_weighted_{t}'] for d in dimensions_names_redux])

    df_paired_shuffled[f'nodimensions_{t}'] = sum([df_paired_shuffled[f'{d}_{t}'] for d in dimensions_names_redux])
    df_paired_shuffled[f'nodimensions_{t}'] = df_paired_shuffled[f'nodimensions_{t}'].apply(lambda x : 0 if x>0 else 1)
    df_paired_shuffled[f'nodimensions_weighted_{t}'] = df_paired_shuffled[f'nodimensions_{t}']
    field_selection = ['post_id', 'comment_id', 'has_delta', 'has_answer_by_op'] 
field_selection += [d+'_post' for d in dimensions_names]
field_selection += [d+'_weighted_post' for d in dimensions_names]
field_selection += ['sentiment_pos_weighted_post', 'sentiment_neg_weighted_post']
field_selection += ['nodimensions_comment', 'nodimensions_post', 'nodimensions_weighted_comment', 'nodimensions_weighted_post']
field_selection += ['anydimensions_comment', 'anydimensions_post', 'anydimensions_weighted_comment', 'anydimensions_weighted_post']
df_posts_delta_nondelta = df_paired[field_selection].drop_duplicates(subset=['post_id', 'comment_id', 'has_delta']) 
df_posts_delta = df_posts_delta_nondelta[df_posts_delta_nondelta['has_delta'] == True].drop_duplicates().dropna()
df_posts_nodelta = df_posts_delta_nondelta[~df_posts_delta_nondelta['post_id'].isin(df_posts_delta['post_id'])].drop_duplicates().dropna()
# WEIGHT DISCOUNTED VERSION

fig, ax = plt.subplots(1, 1, figsize=(7, 6))

plotfile = open('PLOTFILE_odds_ratios_posts_hasanswer.csv', 'wt', encoding='utf8')
plotfile.write('dimension,odds_ratio,ci_95\n')

y_oddsratio = []
y_ratio = []
y_delta = []
y_nodelta = []
y_ci_delta = []
y_ci_nodelta = []
y_ci_OR = []
colors = []

all_dims = dimensions_names_redux + ['sentiment_pos', 'sentiment_neg', 'nodimensions']

for dim in all_dims:

    scores_delta = df_posts_delta[f'{dim}_weighted_post']
    scores_no_delta = df_posts_nodelta[f'{dim}_weighted_post']

    avg_delta, ci_delta = ci(scores_delta)
    avg_no_delta, ci_no_delta = ci(scores_no_delta)

    OR = (avg_delta / (1 - avg_delta)) / (avg_no_delta / (1 - avg_no_delta))
    log_OR = np.log(OR)

    y_oddsratio.append(OR)
    y_ratio.append((avg_delta / avg_no_delta) - 1)
    y_delta.append(avg_delta / avg_no_delta)
    y_nodelta.append(avg_no_delta)
    y_ci_delta.append(ci_delta / avg_no_delta)
    y_ci_nodelta.append(ci_no_delta / avg_no_delta)

    _dim_delta = int(avg_delta * len(scores_delta))
    _no_dim_delta = len(scores_delta) - _dim_delta
    _dim_no_delta = int(avg_no_delta * len(scores_no_delta))
    _no_dim_no_delta = len(scores_no_delta) - _dim_no_delta

    _all_cases = _dim_delta + _no_dim_delta + _dim_no_delta + _no_dim_no_delta

    standard_error_log_OR = np.sqrt(
        1 / _dim_delta
        + 1 / _no_dim_delta
        + 1 / _dim_no_delta
        + 1 / _no_dim_no_delta
    )

    ci_log_OR = 1.96 * standard_error_log_OR

    ci_log_OR_upper = log_OR + ci_log_OR
    ci_log_OR_lower = log_OR - ci_log_OR

    ci_OR_upper = np.exp(ci_log_OR_upper)
    ci_OR_lower = np.exp(ci_log_OR_lower)

    ci_OR = ci_OR_upper - OR
    y_ci_OR.append(ci_OR)

    plotfile.write(f'{dim},{OR},{ci_OR}\n')
    if avg_delta > avg_no_delta:
        clr = 'b'
    else:
        clr = 'r'

    colors.append(clr)

plotfile.close()


# Sort dimensions by odds ratio in descending order
L = list(zip(
    y_oddsratio,
    y_ci_OR,
    y_ratio,
    y_delta,
    y_ci_delta,
    y_nodelta,
    y_ci_nodelta,
    colors,
    all_dims
))

L.sort(key=lambda x: x[0], reverse=True)

labels_ordered = []

for i, l in enumerate(L):

    y_odds, y_odds_ci, y_rat, y_d, err_delta, y_nod, err_nodelta, color, label = l

    labels_ordered.append(label)

    # Plot odds ratio and confidence interval
    ax.errorbar(
        y_odds,
        i,
        xerr=y_odds_ci,
        elinewidth=1,
        fmt='o',
        markersize=4,
        color=color,
        capsize=3,
        zorder=2
    )

    # Display odds ratio above each point in black, without bold
    ax.annotate(
        f'{y_odds:.2f}',
        (y_odds, i),
        textcoords='offset points',
        xytext=(0, 7),
        ha='center',
        va='bottom',
        fontsize=10,
        color='black'
    )


# Configure axes
ax.set_title('Odds ratio (posts)', size=18)
ax.set_yticks(range(len(labels_ordered)))
ax.set_yticklabels(labels_ordered)
ax.set_xlabel('(b)', size=18)

# Numerical axis at the bottom, from 0.0 to 3.0
ax.set_xlim(0, 3.0)
ax.set_xticks(np.arange(0.0, 3.01, 0.5))
ax.xaxis.tick_bottom()
ax.xaxis.set_label_position('bottom')

# Add vertical gridlines
ax.grid(axis='x', linestyle='-', linewidth=0.7, alpha=0.4, zorder=0)

# Reference line at odds ratio = 1
ax.axvline(1, linestyle='--', linewidth=1, color='gray', zorder=1)

# Keep the highest odds ratios at the top
ax.invert_yaxis()

# Remove top and right borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('odds_ratio_posts.png', dpi=300, bbox_inches='tight')
plt.show()
dimensions_names_extended = dimensions_names_redux + ['nodimensions']#, 'sentiment_pos', 'sentiment_neg']
