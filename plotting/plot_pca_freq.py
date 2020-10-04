import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy import stats
import json

import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

model = 'txl'
all_models = ['txl', 'gpt', 'bert']
COMPUTE_MEANS = False
PLOT_PAIRPLOTS = False
SORT_BY_FIRST_PC = True
PC_COUNT = 3

non_number_cols = {'sent', 'freq', 'noun', 's_freq', 'p_freq'}

SVA_TASKS = ['Simple', 'SubjRelClause', 'SentComp', 'PP', 'ObjRelClauseThat', 'ObjRelClauseNoThat'] # Leave out ShortVpCoord
RA_TASKS = ['Simple', 'SentComp', 'ObjRelClauseThat', 'ObjRelClauseNoThat']
SENTTYPE_TASKS = {'sv_agreement': SVA_TASKS, 'anaphora': RA_TASKS}
SHORTENED_SENTTYPES = [('SV', 'sv_agreement'), ('RA', 'anaphora')]

SVA_LABELS = {'Simple': 'Simple\n(SVA)', 'SubjRelClause': 'Subj. RC\n(SVA)',
              'SentComp': 'Sent. Comp.\n(SVA)', 'PP': 'PP\n(SVA)',
              'ObjRelClauseThat': 'Obj. RC [That]\n(SVA)',
              'ObjRelClauseNoThat': 'Obj. RC [No That]\n(SVA)'}
RA_LABELS = {'Simple': 'Simple\n(RA)', 'SentComp': 'Sent. Comp.\n(RA)',
             'ObjRelClauseThat': 'Obj. RC [That]\n(RA)',
             'ObjRelClauseNoThat': 'Obj. RC [No That]\n(RA)'}
TASK_LABELS = {'SVA': SVA_LABELS, 'RA': RA_LABELS}

def load_and_mean_diffs(model): 
    senttype_dfs = {}
    if not COMPUTE_MEANS: 
        full_df = pd.read_csv('./data/%s_all_means.csv' % (model))
        senttype_dfs['SV'] = pd.read_csv('./data/%s_sv_means.csv' % (model))
        senttype_dfs['RA'] = pd.read_csv('./data/%s_ra_means.csv' % (model))
    else: 
        means = {'sv_agreement': {}, 'anaphora': {}}
        for senttype, tasks in SENTTYPE_TASKS.items(): 
            for task in tasks: 
                differences_filename = '../transformer_evals/%s_%s/differences_data/%s.differences.csv' % (model, senttype, task.lower())
                df = pd.read_csv(differences_filename)
                numbers = df[[c for c in df if c not in non_number_cols]]
                mean_vals = numbers.mean(axis=1, numeric_only=None)
                means[senttype][task] = {df['sent'][i]: mean_vals[i] for i in range(len(df))}
        full_df = pd.DataFrame()
        full_df['Noun'] = df['sent']
        
        for shortened, senttype in SHORTENED_SENTTYPES: 
            senttype_df = pd.DataFrame()
            senttype_df['Noun'] = full_df['Noun']
            for task in SENTTYPE_TASKS[senttype]: 
                colname = '%s_%s' % (shortened, task)
                full_df[colname] = full_df['Noun'].map(means[senttype][task])
                senttype_df[task] = senttype_df['Noun'].map(means[senttype][task])
            senttype_df.drop('Noun', axis=1, inplace=True)
            senttype_df = senttype_df[(senttype_df.T != 0).all()]
            senttype_dfs[shortened] = senttype_df
        full_df = full_df[(full_df.T != 0).all()]
        save_dfs(full_df, senttype_dfs, model)
    
    return full_df, senttype_dfs

def save_dfs(full_df, senttype_dfs, model): 
    full_df.to_csv('./data/%s_all_means.csv' % (model), index=False)
    senttype_dfs['SV'].to_csv('./data/%s_sv_means.csv' % (model), index=False)
    senttype_dfs['RA'].to_csv('./data/%s_ra_means.csv' % (model), index=False)

def plot_pairplots(full_df, senttype_dfs): 
    sv_pairplot = sns.pairplot(senttype_dfs['SV'])
    an_pairplot = sns.pairplot(senttype_dfs['RA'])
    full_pairplot = sns.pairplot(full_df)
    return full_pairplot, sv_pairplot, an_pairplot

def run_PCA(full_df): 
    def center_values(dataset): 
        mean_val = np.mean(dataset, axis=0) # 1 x d average values vector
        A = np.subtract(dataset, mean_val) # M x d of centered values
        return A
    dataset = full_df.drop('Noun', axis=1).to_numpy()
    dataset = center_values(dataset)
    pca = PCA(n_components=10)
    reduced = pca.fit_transform(dataset)
    return pca, reduced

def print_pc_vars(pca): 
    print('NonCumulative explained variance: ')
    print(pca.explained_variance_ratio_)
    print('Cumulative explained variance: ')
    for i, val in enumerate(np.cumsum(pca.explained_variance_ratio_)): 
        print(str(i+1)+': ' + str(val))
    
def load_and_map_freqs(df): 
    word_counts = {}
    with open("../sv_agreement/simple/word_freqs.json.txt", "r") as f: 
        word_counts = json.load(f)
    freqs_dict = {}
    for word in word_counts: 
        freqs_dict[word] = word_counts[word][4]
    
    df['Frequency'] = df['Noun'].map(freqs_dict)
    return df

def run_pipeline(model): 
    full_df, senttype_dfs = load_and_mean_diffs(model)
    if COMPUTE_MEANS: 
        save_dfs(full_df, senttype_dfs, model)
    if PLOT_PAIRPLOTS: 
        full_pairplot, sv_pairplot, an_pairplot = plot_pairplots(full_df, senttype_dfs)
        if COMPUTE_MEANS: 
            full_pairplot.savefig('./%s_full_pairplot.png' % (model))
            sv_pairplot.savefig('./%s_sv_pairplot.png' % (model))
            an_pairplot.savefig('./%s_ra_pairplot.png' % (model))
    pca, reduced = run_PCA(full_df)
    print_pc_vars(pca)

    reduced_df = pd.DataFrame()
    reduced_df['Noun'] = full_df['Noun']
    for i in range(PC_COUNT): 
        reduced_df['PC%d' % (i+1)] = reduced.T[i]
    
    reduced_df = load_and_map_freqs(reduced_df)
    if SORT_BY_FIRST_PC: 
        reduced_df.sort_values('PC1', ascending=False, axis=0,inplace=True)
    reduced_df.to_csv('./data/%s_pca.csv' % (model), index=False)
    return reduced_df

def load_all_dicts(): 
    full_df_dict = {}
    senttype_dfs_dict = {}
    for model in all_models: 
        full_df, senttype_dfs = load_and_mean_diffs(model)
        full_df_dict[model] = full_df
        senttype_dfs_dict[model] = senttype_dfs
    return full_df_dict, senttype_dfs_dict

def get_nouns_list(full_df_dict): 
    nouns_sets = [set(full_df_dict[model]['Noun']) for model in all_models]
    total_set = nouns_sets[0]
    for i in range(1, len(nouns_sets)): 
        total_set = total_set.intersection(nouns_sets[i])
    return list(total_set)

def rename_df_dicts(full_df_dict): 
    full_df_dict['gpt2'] = full_df_dict['gpt']
    del full_df_dict['gpt']
    return full_df_dict

def run_correlation_pipeline(): 
    full_df_dict, senttype_dfs_dict = load_all_dicts()
    total_nouns_list = get_nouns_list(full_df_dict)
    full_df_dict = rename_df_dicts(full_df_dict)
    print(len(total_nouns_list))

    df = pd.DataFrame()
    df['Noun'] = total_nouns_list
    for model in full_df_dict: 
        model_suffix = '_%s' % (model)
        column_names = [c+model_suffix if c!='Noun' else c for c in full_df_dict[model].columns]
        full_df_dict[model].columns = column_names
        # print(df.head())
        df = df.join(full_df_dict[model].set_index('Noun'), on='Noun', rsuffix=model_suffix, how='left')
    
    df = df[(df.T != 0).all()]
    return df

# this function and the code in main to draw the plot from: 
# https://towardsdatascience.com/visualizing-data-with-pair-plots-in-python-f228cf529166
def corr(x, y, **kwargs):
    # Calculate the value
    coef = np.corrcoef(x, y)[0][1]
    # Make the label
    label = r'$\rho$ = ' + str(round(coef, 2))
    
    # Add the label to the plot
    ax = plt.gca()
    ax.annotate(label, xy = (0.2, 0.95), size = 20, xycoords = ax.transAxes)


def draw_corr_plot_diag(df, titles, savename=None): 
    def metrics(x, y, ax):
        slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
        rho_label = 'r = %s' % (str(round(r_value, 2)))
        ax.annotate(rho_label, xy = (0.02, 0.84), size = 48, xycoords = ax.transAxes)
        p_label = 'p = %.1g' % p_value
        ax.annotate(p_label, xy = (0.02, 0.92), size = 48, xycoords = ax.transAxes)

    fig, big_axes = plt.subplots(figsize=(100, 30) , nrows=3, ncols=1) 

    fig.suptitle(titles[0], fontsize=75, ha = 'center', va = 'top')

    for row, big_ax in enumerate(big_axes, start=1):
        #big_ax.set_title(titles[row], fontsize=50, pad = 15)

        # Turn off axis lines and ticks of the big subplot 
        # obs alpha is 0 in RGBA string!
        big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
        # removes the white frame
        big_ax._frameon = False

    i = 0
    row = 1
    # fig, ax = plt.subplots(3,10, figsize=(100,30))
    for model1, model2, color in [('gpt2', 'bert', 'r'), ('gpt2', 'txl', 'g'), ('bert', 'txl', 'b')]:
        for shortened, senttype in SHORTENED_SENTTYPES: 
            category = shortened
            if category=='SV': 
                category = 'SVA'
            for task in SENTTYPE_TASKS[senttype]: 
                i += 1
                ax = fig.add_subplot(3,10,i)
                model1_task = f'{shortened}_{task}_{model1}'
                model2_task = f'{shortened}_{task}_{model2}'
                sns.regplot(x=model1_task, y=model2_task, data=df, ax=ax, color=color)
                metrics(df[model1_task], df[model2_task], ax)
                #ax.set_title(f'{category} {task}')

                for side in ['top', 'left', 'bottom', 'right']:
                    ax.spines[side].set_linewidth(3)

                # only show model-pair label for first task
                if(shortened == 'SV' and task == 'Simple'):
                    ax.set_ylabel(titles[row], fontsize = 60, ha = 'right',
                        rotation = 'horizontal')
                    row += 1
                else:
                    ax.set_ylabel('')

                # only show task label for last model
                if(model1 == 'bert' and model2 == 'txl'):
                    ax.set_xlabel(TASK_LABELS[category][task], fontsize = 60,
                        va = 'top')
                else:
                    ax.set_xlabel('')

    fig.set_facecolor('w')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if savename is not None: 
        plt.savefig(savename)

if __name__=='__main__': 
    # reduced_df = run_pipeline('txl')
    df = run_correlation_pipeline()
    print('ran correlation pipeline')
    
    titles = [
        'Pairwise Comparison of Models on each Grammatical Task',
        'GPT-2\nv. BERT', 
        'GPT-2\nv. T-XL', 
        'BERT\nv. T-XL',
    ]
    draw_corr_plot_diag(df, titles, savename='correlation_plot.png')

    # Create a pair grid instance
    # grid = sns.pairplot(data=df, 
    #         x_vars = ['SV_Simple_gpt2', 'SV_SubjRelClause_gpt2', 'SV_SentComp_gpt2', 'SV_PP_gpt2', 'SV_ObjRelClauseThat_gpt2', 'SV_ObjRelClauseNoThat_gpt2', 'RA_Simple_gpt2', 'RA_SentComp_gpt2', 'RA_ObjRelClauseThat_gpt2', 'RA_ObjRelClauseNoThat_gpt2'], 
    #         y_vars = ['SV_Simple_bert', 'SV_SubjRelClause_bert', 'SV_SentComp_bert', 'SV_PP_bert', 'SV_ObjRelClauseThat_bert', 'SV_ObjRelClauseNoThat_bert', 'RA_Simple_bert', 'RA_SentComp_bert', 'RA_ObjRelClauseThat_bert', 'RA_ObjRelClauseNoThat_bert'], 
    #         kind='reg',
    #         size = 4)
    # grid = grid.map(corr)
    # grid.savefig('./gpt_bert_correlation_plot.png')

    # grid = sns.pairplot(data=df, 
    #         x_vars = ['SV_Simple_gpt2', 'SV_SubjRelClause_gpt2', 'SV_SentComp_gpt2', 'SV_PP_gpt2', 'SV_ObjRelClauseThat_gpt2', 'SV_ObjRelClauseNoThat_gpt2', 'RA_Simple_gpt2', 'RA_SentComp_gpt2', 'RA_ObjRelClauseThat_gpt2', 'RA_ObjRelClauseNoThat_gpt2'], 
    #         y_vars = ['SV_Simple_txl', 'SV_SubjRelClause_txl', 'SV_SentComp_txl', 'SV_PP_txl', 'SV_ObjRelClauseThat_txl', 'SV_ObjRelClauseNoThat_txl', 'RA_Simple_txl', 'RA_SentComp_txl', 'RA_ObjRelClauseThat_txl', 'RA_ObjRelClauseNoThat_txl'], 
    #         kind='reg',
    #         size = 4)
    # grid = grid.map(corr)
    # grid.savefig('./gpt_txl_correlation_plot.png')
    

    # grid = sns.pairplot(data=df, 
    #         x_vars = ['SV_Simple_bert', 'SV_SubjRelClause_bert', 'SV_SentComp_bert', 'SV_PP_bert', 'SV_ObjRelClauseThat_bert', 'SV_ObjRelClauseNoThat_bert', 'RA_Simple_bert', 'RA_SentComp_bert', 'RA_ObjRelClauseThat_bert', 'RA_ObjRelClauseNoThat_bert'], 
    #         y_vars = ['SV_Simple_txl', 'SV_SubjRelClause_txl', 'SV_SentComp_txl', 'SV_PP_txl', 'SV_ObjRelClauseThat_txl', 'SV_ObjRelClauseNoThat_txl', 'RA_Simple_txl', 'RA_SentComp_txl', 'RA_ObjRelClauseThat_txl', 'RA_ObjRelClauseNoThat_txl'], 
    #         kind='reg',
    #         size = 4)
    # grid = grid.map(corr)
    # grid.savefig('./bert_txl_correlation_plot.png')

    # grid = sns.PairGrid(data=df, vars = [c for c in df.columns if c!='Noun'], size = 4)

    # Map the plots to the locations
    # grid = grid.map_upper(plt.scatter, color = 'Green')
    # grid = grid.map_upper(corr)
    # grid = grid.map_lower(sns.kdeplot, cmap = 'Reds')
    # grid = grid.map_diag(plt.hist, bins = 10, edgecolor =  'k', color = 'Blue')
    # grid.savefig('./correlation_plot.png')
