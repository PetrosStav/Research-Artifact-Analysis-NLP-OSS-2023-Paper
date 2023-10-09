import pandas as pd

def remove_percentages(string):
    if '%' in string:
        return string.split(' ')[0].strip()
    else:
        return string
    
table_header = r"""
\begin{table*}[!htbp]
    \centering
    \tiny
    \begin{tabular}{@{\hspace{3pt}}c@{\hspace{3pt}}c@{\hspace{3pt}}c@{\hspace{3pt}}c@{\hspace{3pt}}c@{\hspace{3pt}}c@{\hspace{3pt}}c@{\hspace{3pt}}c@{\hspace{3pt}}c@{\hspace{3pt}}c||@{\hspace{3pt}}@{\hspace{3pt}}c@{\hspace{3pt}}c@{\hspace{3pt}}c@{\hspace{3pt}}c@{\hspace{3pt}}c@{\hspace{3pt}}c@{\hspace{3pt}}c@{\hspace{3pt}}c@{\hspace{3pt}}c@{\hspace{3pt}}c@{\hspace{3pt}}}
    \hline
        ~ & ~ & ~ & ~ & \textbf{Original} & ~ & ~ & ~ & ~ & ~ & ~ & ~ & ~ & \textbf{Augmented} & ~ & ~ & ~ & ~ &   \\ \hline
        ~ & \textbf{Train} & ~ & ~ & \textbf{Dev} & ~ & ~ & \textbf{Test} & ~ & ~ & \textbf{Train} & ~ & ~ & \textbf{Dev} & ~ & ~ & \textbf{Test} & ~ &   \\ \hline
        ~ & \textbf{dataset} & \textbf{software} & \textbf{all} & \textbf{dataset} & \textbf{software} & \textbf{all} & \textbf{dataset} & \textbf{software} & \textbf{all} ~ & \textbf{dataset} & \textbf{software} & \textbf{all} & \textbf{dataset} & \textbf{software} & \textbf{all} & \textbf{dataset} & \textbf{software} & \textbf{all}  \\ \hline
        """

table_footer = """
\end{tabular}
    \caption{Statistics for the Synthetic dataset.}
    \label{tab:synthetic_stats}
\end{table*}"""

print('SYNTHETIC DATASET - Snippets')
synthetic_og_snippets = pd.read_excel('./material/synthetic_dataset_og_stats_snippets.xlsx').to_dict('records')
synthetic_aug_snippets = pd.read_excel('./material/synthetic_dataset_aug_stats_snippets.xlsx').to_dict('records')

print(table_header)
for line, line2 in zip(synthetic_og_snippets[2:], synthetic_aug_snippets[2:]):
    line_values = list(line.values())
    line2_values = list(line2.values())
    print('\\textbf{' + line_values[0] + '} & ' + ' & '.join([remove_percentages(str(x)) for x in line_values[1:]]) + ' & ' + ' & '.join([remove_percentages(str(x)) for x in line2_values[1:]]) + ' \\\\')
print(table_footer)

print()

print('SYNTHETIC DATASET - QA Pairs')
synthetic_og_qa_pairs = pd.read_excel('./material/synthetic_dataset_og_stats_qa_pairs.xlsx').to_dict('records')
synthetic_aug_qa_pairs = pd.read_excel('./material/synthetic_dataset_aug_stats_qa_pairs.xlsx').to_dict('records')

print(table_header)
for line, line2 in zip(synthetic_og_qa_pairs[2:], synthetic_aug_qa_pairs[2:]):
    line_values = list(line.values())
    line2_values = list(line2.values())
    print('\\textbf{' + line_values[0] + '} & ' + ' & '.join([remove_percentages(str(x)) for x in line_values[1:]]) + ' & ' + ' & '.join([remove_percentages(str(x)) for x in line2_values[1:]]) + ' \\\\')
print(table_footer)

print()

print('HYBRID DATASET - Snippets')
hybrid_og_snippets = pd.read_excel('./material/hybrid_dataset_og_stats_snippets.xlsx').to_dict('records')
hybrid_aug_snippets = pd.read_excel('./material/hybrid_dataset_aug_stats_snippets.xlsx').to_dict('records')

print(table_header)
for line, line2 in zip(hybrid_og_snippets[2:], hybrid_aug_snippets[2:]):
    line_values = list(line.values())
    line2_values = list(line2.values())
    print('\\textbf{' + line_values[0] + '} & ' + ' & '.join([remove_percentages(str(x)) for x in line_values[1:]]) + ' & ' + ' & '.join([remove_percentages(str(x)) for x in line2_values[1:]]) + ' \\\\')
print(table_footer)

print()

print('HYBRID DATASET - QA Pairs')
hybrid_og_qa_pairs = pd.read_excel('./material/hybrid_dataset_og_stats_qa_pairs.xlsx').to_dict('records')
hybrid_aug_qa_pairs = pd.read_excel('./material/hybrid_dataset_aug_stats_qa_pairs.xlsx').to_dict('records')

print(table_header)
for line, line2 in zip(hybrid_og_qa_pairs[2:], hybrid_aug_qa_pairs[2:]):
    line_values = list(line.values())
    line2_values = list(line2.values())
    print('\\textbf{' + line_values[0] + '} & ' + ' & '.join([remove_percentages(str(x)) for x in line_values[1:]]) + ' & ' + ' & '.join([remove_percentages(str(x)) for x in line2_values[1:]]) + ' \\\\')
print(table_footer)

print()
