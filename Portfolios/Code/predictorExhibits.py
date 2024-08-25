import os
import requests
import zipfile
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import matplotlib.transforms as transforms
import matplotlib.patheffects as path_effects

import math
from adjustText import adjust_text
import statsmodels.api as sm
from tabulate import tabulate

from globals import pathDataIntermediate, pathResults, pathDataPortfolios, pathProject

url = "http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
dest = f'{pathDataIntermediate}temp.zip'


# Download the file
response = requests.get(url)
with open(dest, 'wb') as file:
    file.write(response.content)

# Unzip the file
with zipfile.ZipFile(dest, 'r') as zip_ref:
    zip_ref.extractall(pathDataIntermediate)

# Load FF 5 factors
ff = pd.read_csv(
    os.path.join(pathDataIntermediate, "F-F_Research_Data_5_Factors_2x3.csv"),
    skiprows=3
).rename(columns={ff.columns[0]: 'date'})

ff['date'] = pd.to_datetime(ff['date'].astype(str) + '28', format='%Y%m%d')

# Load Momentum Factor
momentum = pd.read_csv(
    os.path.join(pathDataIntermediate, "F-F_Momentum_Factor.csv"),
    skiprows=13
).rename(columns={momentum.columns[0]: 'date'})
momentum['date'] = pd.to_datetime(momentum['date'].astype(str) + '28', format='%Y%m%d')

# Merge datasets
ff = ff.merge(momentum, on='date', how='outer')
ff = ff.sort_values('date')

# Figure 1 (Port): Correlations (Portfolio level) -------------------------

## import baseline returns (clear/likely)
retwide = pd.read_csv(os.path.join(pathDataPortfolios, "PredictorLSretWide.csv"))
retwide['date'] = pd.to_datetime(retwide['date'].str[:8] + '28', format='%Y%m%d')

# Fig 1a: Pairwise correlation of strategy returns
corr_matrix = retwide.drop(columns=['date']).corr(method='pearson')
lower_tri_corr = corr_matrix.where(np.tril(np.ones(corr_matrix.shape), k=-1).astype(bool))
temp = lower_tri_corr.stack().reset_index(drop=True)


df = pd.DataFrame({'rho': temp})

plt.figure(figsize=(10, 8))  # Set the figure size to 10x8 inches
sns.histplot(df['rho'], bins=30)
plt.xlim(-1, 1)
plt.xlabel("Pairwise correlation coefficient")
plt.ylabel("Count")
sns.set_theme(style="whitegrid")

# Save the plot as a PDF file
plt.savefig(os.path.join(pathResults, "fig1aPort_pairwisecorrelations.pdf"), format='pdf')

allRhos = pd.DataFrame({
    'rho': temp,
    'series': 'Pairwise'
})

tempRets = pd.merge(retwide, ff[['date', 'SMB']], on='date', how='inner')
temp = tempRets.drop(columns=['date', 'SMB']).corrwith(tempRets['SMB'], method='pearson')
rho_df = pd.DataFrame({'rho': temp})
plt.figure(figsize=(10, 8))
sns.histplot(rho_df['rho'], bins=25)
plt.xlim(-1, 1)
plt.xlabel("Pairwise correlation coefficient")
plt.ylabel("Count")
sns.set_theme(style="whitegrid")
plt.savefig(os.path.join(pathResults, "fig1bPort_correlationSMB.pdf"), format='pdf')


new_rhos = pd.DataFrame({
    'rho': temp,
    'series': 'Size'
})


allRhos = pd.concat([allRhos, new_rhos], ignore_index=True)
tempRets = pd.merge(retwide, ff[['date', 'HML']], on='date', how='inner')
temp = tempRets.drop(columns=['date', 'HML']).corrwith(tempRets['HML'], method='pearson')
rho_df = pd.DataFrame({'rho': temp})

plt.figure(figsize=(10, 8))
sns.histplot(rho_df['rho'], bins=25)
plt.xlim(-1, 1)
plt.xlabel("Pairwise correlation coefficient")
plt.ylabel("Count")
sns.set_theme(style="whitegrid")

# Save the plot as a PDF
plt.savefig(os.path.join(pathResults, "fig1cPort_correlationHML.pdf"), format='pdf')

new_rhos = pd.DataFrame({
    'rho': temp,
    'series': 'Value'
})
allRhos = pd.concat([allRhos, new_rhos], ignore_index=True)

# Fig 1d: Correlation with Mom
tempRets = pd.merge(retwide, ff[['date', 'Mom']], on='date', how='inner')
temp = tempRets.drop(columns=['date', 'Mom']).corrwith(tempRets['Mom'], method='pearson')

rho_df = pd.DataFrame({'rho': temp})
plt.figure(figsize=(10, 8))
sns.histplot(rho_df['rho'], bins=25)
plt.xlim(-1, 1)
plt.xlabel("Pairwise correlation coefficient")
plt.ylabel("Count")
sns.set_theme(style="whitegrid")
# Save the plot as a PDF
plt.savefig(os.path.join(pathResults, "fig1dPort_correlationMom.pdf"), format='pdf')


new_rhos = pd.DataFrame({
    'rho': temp,
    'series': 'Momentum'
})
allRhos = pd.concat([allRhos, new_rhos], ignore_index=True)

# Fig 1e: Correlation with OP
tempRets = pd.merge(retwide, ff[['date', 'RMW']], on='date', how='inner')
temp = tempRets.drop(columns=['date', 'RMW']).corrwith(tempRets['RMW'], method='pearson')
rho_df = pd.DataFrame({'rho': temp})

# Plot the histogram
plt.figure(figsize=(10, 8))
sns.histplot(rho_df['rho'], bins=25)
plt.xlim(-1, 1)
plt.xlabel("Pairwise correlation coefficient")
plt.ylabel("Count")
sns.set_theme(style="whitegrid")
# Save the plot as a PDF
plt.savefig(os.path.join(pathResults, "fig1ePort_correlationOP.pdf"), format='pdf')

new_rhos = pd.DataFrame({
    'rho': temp,
    'series': 'Profitability'
})
allRhos = pd.concat([allRhos, new_rhos], ignore_index=True)

# Fig 1f: Correlation with Inv
tempRets = pd.merge(retwide, ff[['date', 'CMA']], on='date', how='inner')
temp = tempRets.drop(columns=['date', 'CMA']).corrwith(tempRets['CMA'], method='pearson')

rho_df = pd.DataFrame({'rho': temp})

# Plot the histogram
plt.figure(figsize=(10, 8))
sns.histplot(rho_df['rho'], bins=25)
plt.xlim(-1, 1)
plt.xlabel("Pairwise correlation coefficient")
plt.ylabel("Count")
sns.set_theme(style="whitegrid")
# Save the plot as a PDF
plt.savefig(os.path.join(pathResults, "fig1fPort_correlationCMA.pdf"), format='pdf')

new_rhos = pd.DataFrame({
    'rho': temp,
    'series': 'Investment'
})
allRhos = pd.concat([allRhos, new_rhos], ignore_index=True)

# Print all correlations together
allRhos['series'] = pd.Categorical(allRhos['series'], 
                                   categories=["Pairwise", "Size", "Value", "Momentum", "Profitability", "Investment"],
                                   ordered=True)
# Plot the histogram
plt.figure(figsize=(10, 8))
g = sns.FacetGrid(allRhos, col="series", col_wrap=2, sharey=False)
g.map(sns.histplot, "rho", bins=25)
g.set_axis_labels("Correlation coefficient", "Count")
# Set the x-axis limits
for ax in g.axes.flat:
    ax.set_xlim(-1, 1)

# Apply minimal theme
sns.set_theme(style="whitegrid")

# Save the plot as a PDF
plt.savefig(os.path.join(pathResults, "fig1Port_jointly.pdf"), format='pdf')

# Figure 2: Replication rates ---------------------------------------------

df0 = pd.read_excel(os.path.join(pathDataPortfolios, "PredictorSummary.xlsx"), sheet_name='short')
df0['success'] = (df0['tstat'].round(2) >= 1.96).astype(int)
df0 = df0[['signalname', 'tstat', 'success', 'T.Stat', 'samptype']]

# Check if all rows have 'samptype' equal to 'insamp'
if (df0['samptype'].eq('insamp').sum() != len(df0)):
    print('Mixing different sample types below!!')

# Use most recent Category labels and keep comparable predictors only

df_meta = readdocumentation()
df_meta['comparable'] = (
    (df_meta['Cat.Signal'] == 'Predictor') &
    (df_meta['Predictability.in.OP'] == '1_clear') &
    (df_meta['Signal.Rep.Quality'] != '4_lack_data')
)
df_meta = df_meta[['signalname', 'Cat.Data', 'comparable', 'Predictability.in.OP', 'Signal.Rep.Quality']]
df_meta['Cat.Data'] = df_meta['Cat.Data'].replace('Options', 'Other')

# Select specific columns from df0
df = df0[['signalname', 'success', 'tstat']]
df = pd.merge(df, df_meta, on='signalname', how='left')
df = df[df['comparable']]

# Replication success by data category (for baseline ones)
labelData = df.groupby('Cat.Data').agg(
    rate=('success', 'mean'),
    n=('success', 'size')
).reset_index()

labelData['rate'] = (labelData['rate'] * 100).round(0).astype(int).astype(str) + '%'

# Group by 'Cat.Data' and 'success', and count occurrences
df_grouped = df.groupby(['Cat.Data', 'success']).size().reset_index(name='n')

# Reverse the order of 'Cat.Data' and relevel 'Other' to be the first
df_grouped['Cat.Data'] = pd.Categorical(df_grouped['Cat.Data'], categories=df_grouped['Cat.Data'].unique()[::-1], ordered=True)

# Create the plot
plt.figure(figsize=(10, 8))
sns.barplot(
    data=df_grouped, 
    x='Cat.Data', y='n', 
    hue=pd.Categorical(df_grouped['success'], categories=[0, 1], ordered=True).rename_categories(["No", "Yes"]),
    palette=['gray75', 'gray45']
)

# Add labels and customize plot
plt.xlabel("Data Category")
plt.ylabel("Number of predictors")
plt.legend(title="|t-stat| > 1.96", loc='upper right', reversed=True)
plt.xticks(rotation=90)
plt.gca().invert_yaxis()  # To mimic coord_flip() in R

# Add text labels for rate from labelData
for i, row in labelData.iterrows():
    plt.text(
        row.name, row['n'] + 5,  # Position text above bars
        row['rate'], ha='center'
    )

# Apply minimal theme
sns.set_theme(style="whitegrid")

# Save the plot as a PDF
plt.savefig(os.path.join(pathResults, "fig2b_reprate_data.pdf"), format='pdf')


# Alternatively: Jitter plot
df_plot = df.copy()
df_plot['tstat'] = df_plot['tstat'].abs()


df_plot['Cat.Data'] = pd.Categorical(df_plot['Cat.Data'], categories=df_plot['Cat.Data'].unique()[::-1], ordered=True)

# Create the plot
plt.figure(figsize=(10, 8))
sns.stripplot(
    data=df_plot, x='Cat.Data', y='tstat', 
    jitter=0.15, alpha=0.7
)

plt.axhline(y=1.96, color='gray', linestyle='--')

plt.xlabel("Predictor Category")
plt.ylabel("t-statistic")
plt.xticks(rotation=90)
plt.gca().invert_yaxis() 

sns.set_theme(style="whitegrid")

# Save the plot as a PDF
plt.savefig(os.path.join(pathResults, "fig2b_reprate_data_Jitter.pdf"), format='pdf')

# Scatter of replication t-stat vs OP t-stat ------------------------------
docnew = readdocumentation() # for easy updating of documentation

df = pd.read_excel(
    os.path.join(pathDataPortfolios, "PredictorSummary.xlsx"),
    sheet_name='full'
)
df_filtered = df[(df['samptype'] == 'insamp') & (df['port'] == 'LS')]

df_filtered = df_filtered[['signalname', 'tstat']]

df_joined = df_filtered.merge(docnew, on='signalname', how='left')

df_transformed = df_joined.assign(
    tstatRep=df_joined['tstat'].abs(),
    tstatOP=df_joined['T.Stat'].astype(float).abs(),
    PredOP=df_joined['Predictability.in.OP'],
    RepType=df_joined['Signal.Rep.Quality'],
    OPTest=df_joined['Test.in.OP'],
    Evidence_Summary=df_joined['Evidence.Summary']
)

df_transformed = df_transformed.assign(
    porttest=(
        df_transformed['OPTest'].str.contains('port sort', case=False, na=False) |
        df_transformed['OPTest'].str.contains('LS', case=False, na=False) |
        df_transformed['OPTest'].str.contains('double sort', case=False, na=False)
    ),
    standard=~(
        df_transformed['OPTest'].str.contains('nonstandard', case=False, na=False) |
        df_transformed['OPTest'].str.contains('FF3 style', case=False, na=False)
    )
)
df_final = df_transformed[
    df_transformed['PredOP'].isin(['1_clear', '2_likely'])
]

# Filter the DataFrame to select comparable rows
df_plot = df_final[
    (~df_final['OPTest'].isna()) &  # Ensure 'OPTest' is not missing
    (df_final['porttest']) &        # Only include rows where 'porttest' is True
    (df_final['RepType'].isin(['1_good', '2_fair']))  # Include only specific 'RepType' values
]

X = df_plot['tstatOP']
y = df_plot['tstatRep']

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

summary = model.summary()
intercept = model.params['const']
slope = model.params['tstatOP']
r_squared = model.rsquared
regstr = f"[t reproduction] = {round(intercept, 2)} + {round(slope, 2):.2f} [t original], R-sq = {round(100 * r_squared, 0)}%"


ablines = pd.DataFrame({
    'slope': [1, round(slope, 2)],
    'intercept': [0, round(intercept, 2)],
    'group': pd.Categorical(
        ['45 degree line', 'OLS fit'],
        categories=['OLS fit', '45 degree line']
    )
})
df_plot['PredOP'] = pd.Categorical(df_plot['PredOP'],
                                   categories=['1_clear', '2_likely', '4_not'],
                                   ordered=True)
df_plot['PredOP'] = df_plot['PredOP'].cat.rename_categories(['Clear', 'Likely', 'Not'])


plt.figure(figsize=(10, 8))
sns.scatterplot(data=df_plot, x='tstatOP', y='tstatRep', hue='PredOP', style='PredOP', s=100,
                markers={"Clear": 'o', "Likely": 's', "Not": 'D'}, palette="deep")

for _, row in ablines.iterrows():
    plt.plot([1.5, 17], 
             [row['intercept'] + row['slope'] * 1.5, row['intercept'] + row['slope'] * 17], 
             linestyle='--' if row['group'] == '45 degree line' else '-', 
             label=row['group'])

plt.text(3.3, 14, regstr, fontsize=12, ha='center', path_effects=[path_effects.withStroke(linewidth=3, foreground="white")])

plt.xscale('log')
plt.yscale('log')

plt.xlim(1.5, 17)
plt.ylim(1.0, 15)

plt.xlabel('t-stat original study', fontsize=14)
plt.ylabel('t-stat reproduction', fontsize=14)

plt.xticks([2, 5, 10, 15], ['2', '5', '10', '15'])
plt.yticks([2, 5, 10, 15], ['2', '5', '10', '15'])
plt.legend(title='Predictor Category', loc='upper left')

sns.despine()
plt.grid(True, which="both", ls="--")
texts = [plt.text(x, y, label) for x, y, label in zip(df_plot['tstatOP'], df_plot['tstatRep'], df_plot['signalname'])]
adjust_text(texts, arrowprops=dict(arrowstyle="-", color='k', lw=0.5))

temp = 1.5
plt.savefig(f"{pathResults}/fig_tstathand_vs_tstatOP.pdf", 
            width=10 * temp, 
            height=6 * temp)


filtered_df = df_plot[(df_plot['PredOP'] == '2_likely') & (df_plot['tstatOP'].notna())]
result_df = filtered_df[['signalname', 'Authors', 'tstatRep', 'tstatOP', 'RepType']].sort_values(by='Authors')
filtered_df = df[(df['porttest']) & 
                 (df['tstatOP'].notna()) & 
                 (~df['RepType'].isin(['1_good', '2_fair']))]

result_df = filtered_df[['signalname', 'Authors', 'tstatRep', 'tstatOP', 'RepType']].sort_values(by='Authors')
df_plot['PredOP'] = pd.Categorical(
    df_plot['PredOP'], 
    categories=['1_clear', '2_likely', '4_not'], 
    ordered=True
).rename_categories({'1_clear': 'Clear', '2_likely': 'Likely', '4_not': 'Not'})

df_plot['color'] = df_plot['signalname'].isin(["TrendFactor", "Recomm_ShortInterest"])

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

sns.scatterplot(
    data=df_plot, 
    x='tstatOP', y='tstatRep', hue='color', 
    style='PredOP', markers={19: 'o', 2: 's', 3: '^'}, 
    s=100, ax=ax, palette={True: "#e41a1c", False: "#000000"}, legend=False
)

ax.plot([1.5, 17], [summary.params[0] + summary.params[1]*1.5, summary.params[0] + summary.params[1]*17], 
        linestyle='-', color='black', label='OLS fit')
ax.plot([1.5, 17], [1.5, 17], linestyle='--', color='grey', label='45 degree line')

ax.text(3.3, 14, regstr, fontsize=8)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1.5, 17)
ax.set_ylim(1.0, 15)
ax.set_xlabel('t-stat original study')
ax.set_ylabel('t-stat reproduction')

ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.xaxis.set_minor_formatter(ticker.NullFormatter())
ax.yaxis.set_minor_formatter(ticker.NullFormatter())
ax.set_xticks([2, 5, 10, 15])
ax.set_yticks([2, 5, 10, 15])

texts = []
for line in range(0, df_plot.shape[0]):
    texts.append(ax.text(
        df_plot.tstatOP.iloc[line], 
        df_plot.tstatRep.iloc[line], 
        df_plot.signalname.iloc[line],
        fontsize=8
    ))
adjust_text(texts, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

legend_elements = [
    Line2D([0], [0], color='#e41a1c', lw=4, label='Highlighted Signals'),
    Line2D([0], [0], color='#000000', lw=4, label='Other Signals'),
    Line2D([0], [0], marker='o', color='w', label='Clear', markerfacecolor='black', markersize=10),
    Line2D([0], [0], marker='s', color='w', label='Likely', markerfacecolor='black', markersize=10),
    Line2D([0], [0], marker='^', color='w', label='Not', markerfacecolor='black', markersize=10)
]
ax.legend(handles=legend_elements, loc='upper right', frameon=False)

plt.show()

# Save the plot
fig.savefig(f"{pathResults}/fig_tstathand_vs_tstatOP_withHighlights.pdf", 
            width=10*temp, height=6*temp)

# McLean and Pontiff style graphs -----------------------------------------

# stats
path = f"{pathDataPortfolios}/PredictorSummary.xlsx"
stats = pd.read_excel(path, sheet_name='short')
stats = stats[['signalname', 'tstat', 'rbar']]

csv_path = f"{pathProject}/Comparison_to_MetaReplications.csv"
mpSignals = pd.read_csv(csv_path)
mpSignals_filtered = mpSignals[(mpSignals['metastudy'] == 'MP') & (mpSignals['ourname'] != '_missing_')]

# Merge data
# alldocumentation is created in 00_SettingsAndTools.R
df_documentation = readdocumentation()
df_documentation['inMP'] = df_documentation['signalname'].isin(mpSignals_filtered['ourname'])
df_merge = df_documentation[(df_documentation['Cat.Signal'] == 'Predictor') | (df_documentation['inMP'])]
df_merge = df_merge.merge(stats, on='signalname', how='left')
statsFull_transformed = statsFull.rename(columns={'tstat': 'tstatPS', 'rbar': 'rbarPS'})
df_merge = df_merge.merge(statsFull_transformed[['signalname', 'tstatPS', 'rbarPS']], on='signalname', how='left')
for col in ['tstat', 'tstatPS', 'rbar', 'rbarPS']:
    df_merge[col] = df_merge[col].apply(lambda x: abs(x) if x < 0 else x)

df_merge['DeclineTstat'] = df_merge['tstat'] - df_merge['tstatPS']
df_merge['DeclineRBar'] = df_merge['rbar'] - df_merge['rbarPS']

# Map the 'Category' values to the appropriate labels
category_map = {
    'indirect': 'no evidence',
    '4_not': 'not',
    '3_maybe': 'maybe',
    '2_likely': 'likely',
    '1_clear': 'clear'
}
df_merge['Category'] = df_merge['Predictability.in.OP'].map(category_map)
df_merge = df_merge[['signalname', 'tstat', 'tstatPS', 'DeclineTstat', 'rbar', 'rbarPS', 'DeclineRBar', 'Category', 'Cat.Signal', 'inMP']]
df_merge = df_merge.rename(columns={'Cat.Signal': 'CatPredPlacebo'})
df_merge = df_merge[(df_merge['signalname'] != 'IO_ShortInterest') & df_merge['Category'].isin(['clear', 'likely'])]

# In-sample return
df_merge['inMPStr'] = df_merge['inMP'].apply(lambda x: 'in MP (2016)' if x else 'not in MP (2016)')
plt.figure(figsize=(10, 8))
sns.set(style="whitegrid")
sns.regplot(data=df_merge, x='DeclineRBar', y='rbar', 
            scatter_kws={'s': 50, 'alpha': 0.6, 'edgecolor': 'k'}, 
            line_kws={'color': 'black'}, ci=None)
sns.scatterplot(data=df_merge, x='DeclineRBar', y='rbar', 
                hue='inMPStr', style='inMPStr', markers=["o", "s"], 
                palette=["black", "grey"], s=100)
plt.plot([-1.0, 2], [-1.0, 2], linestyle='dotted', color='grey')
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.xlabel('Decline in return post-publication', fontsize=14)
plt.ylabel('In-Sample return', fontsize=14)
plt.legend(title='', loc='upper right')
plt.xlim(-1.0, 2)
plt.ylim(0, 2.5)
plt.title('')
sns.despine()

# In-sample t-stat
df_merge['inMPStr'] = df_merge['inMP'].apply(lambda x: 'in MP (2016)' if x else 'not in MP (2016)')
plt.figure(figsize=(10, 8))
sns.set(style="whitegrid")
sns.regplot(data=df_merge, x='DeclineRBar', y='tstat', 
            scatter_kws={'s': 50, 'alpha': 0.6, 'edgecolor': 'k'}, 
            line_kws={'color': 'black'}, ci=None)
sns.scatterplot(data=df_merge, x='DeclineRBar', y='tstat', 
                hue='inMPStr', style='inMPStr', markers=["o", "s"], 
                palette=["black", "grey"], s=100)
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.xlabel('Decline in return post-publication', fontsize=14)
plt.ylabel('In-Sample t-statistic', fontsize=14)
plt.legend(title='', loc='upper left')
plt.xlim(-1.0, 2)
plt.ylim(0, 14)
plt.yticks(range(0, 15, 2))
plt.title('')
sns.despine()

#join plots together
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(7, 8))
sns.regplot(data=df_merge, x='DeclineRBar', y='rbar', ax=ax1, 
            scatter_kws={'s': 50, 'alpha': 0.6, 'edgecolor': 'k'}, 
            line_kws={'color': 'black'}, ci=None)
sns.scatterplot(data=df_merge, x='DeclineRBar', y='rbar', ax=ax1,
                hue='inMPStr', style='inMPStr', markers=["o", "s"], 
                palette=["black", "grey"], s=100)
ax1.axhline(0, color='black', linewidth=1)
ax1.axvline(0, color='black', linewidth=1)
ax1.set_xlabel('Decline in return post-publication')
ax1.set_ylabel('In-Sample return')
ax1.legend(title='', loc='upper left')
ax1.set_xlim(-1.0, 2)
ax1.set_ylim(0, 2.5)
sns.regplot(data=df_merge, x='DeclineRBar', y='tstat', ax=ax2, 
            scatter_kws={'s': 50, 'alpha': 0.6, 'edgecolor': 'k'}, 
            line_kws={'color': 'black'}, ci=None)
sns.scatterplot(data=df_merge, x='DeclineRBar', y='tstat', ax=ax2,
                hue='inMPStr', style='inMPStr', markers=["o", "s"], 
                palette=["black", "grey"], s=100)
ax2.axhline(0, color='black', linewidth=1)
ax2.axvline(0, color='black', linewidth=1)
ax2.set_xlabel('Decline in return post-publication')
ax2.set_ylabel('In-Sample t-statistic')
ax2.legend(title='', loc='upper left')
ax2.set_xlim(-1.0, 2)
ax2.set_ylim(0, 14)
ax2.set_yticks(range(0, 15, 2))

# Adjust layout
plt.tight_layout()

# Save the combined plot as a PDF
plt.savefig(f'{pathResults}fig5_MP_both.pdf')

# manual inspection 
result = df_merge[df_merge['inMP']].loc[:, ['signalname', 'tstat', 'Category']].sort_values(by='tstat')
result = df_merge[df_merge['inMP']].agg({
    'rbar': ['mean', 'std'],  # Calculate mean and standard deviation of 'rbar'
    'tstat': lambda x: (x > 1.5).sum()  # Count the number of rows where 'tstat' > 1.5
})


# Big summary table for paper ---------------------------------------------

stats = pd.read_excel(
    io=pathDataPortfolios + "PredictorSummary.xlsx", 
    sheet_name='short'
)
statsFull = pd.read_excel(
    io=pathDataPortfolios + "PredictorSummary.xlsx", 
    sheet_name='full'
)
# Merge data
# alldocumentation is created in 00_SettingsAndTools.R
df_merge = readdocumentation.copy()
df_merge = df_merge[df_merge['Cat.Signal'] == 'Predictor']
df_merge = df_merge.merge(
    stats[['signalname', 'tstat', 'rbar']], 
    on='signalname', 
    how='left'
)
statsFull_filtered = statsFull[(statsFull['samptype'] == 'postpub') & (statsFull['port'] == 'LS')]
statsFull_filtered = statsFull_filtered[['signalname', 'tstat']].rename(columns={'tstat': 't-stat PS'})

df_merge = df_merge.merge(
    statsFull_filtered, 
    on='signalname', 
    how='left'
)
df_merge['ref'] = df_merge['Authors'] + ' (' + df_merge['Year'].astype(str) + ')'
df_merge['Predictor'] = df_merge['LongDescription']
df_merge['sample'] = df_merge['SampleStartYear'].astype(str) + '-' + df_merge['SampleEndYear'].astype(str)
df_merge['Mean Return'] = df_merge['rbar'].round(2)
df_merge['t-stat IS'] = df_merge['tstat'].round(2)
df_merge['Evidence'] = df_merge['Evidence.Summary']
category_mapping = {
    "indirect": "no evidence",
    "4_not": "not",
    "3_maybe": "maybe",
    "2_likely": "likely",
    "1_clear": "clear"
}

df_merge['Category'] = df_merge['Predictability.in.OP'].map(category_mapping)
df_merge = df_merge[['ref', 'Predictor', 'signalname', 'sample', 'Mean Return', 't-stat IS', 'Evidence', 'Category']]
df_merge = df_merge.sort_values(by='ref').reset_index(drop=True)

# Create Latex output table 1: Clear Predictors
clear_predictors = df_merge[df_merge['Category'] == "clear"].copy()
clear_predictors = clear_predictors.drop(columns=['Category'])
latex_table = clear_predictors.to_latex(index=False, column_format='|c|c|c|c|c|c|c|', longtable=False)

# Save to a .tex file
with open("clear_predictors_table.tex", "w") as file:
    file.write(latex_table)

latex_table = tabulate(clear_predictors, headers='keys', tablefmt='latex', showindex=False)

file_path = "bigSignalTableClear.tex" 
with open(file_path, "w") as file:
    file.write(latex_table)

likely_predictors = df_merge[df_merge['Category'] == "likely"].drop(columns=['Category'])
latex_table_likely = tabulate(likely_predictors, headers='keys', tablefmt='latex', showindex=False)
file_path_likely = "bigSignalTableLikely.tex"
with open(file_path_likely, "w") as file:
    file.write(latex_table_likely)

likely_predictors = df_merge[df_merge['Category'] == "likely"].drop(columns=['Category'])
latex_table_likely = tabulate(likely_predictors.values.tolist(), tablefmt='latex', showindex=False)
file_path_likely = f"{pathResults}/bigSignalTableLikely.tex"
with open(file_path_likely, "w") as file:
    file.write(latex_table_likely)
