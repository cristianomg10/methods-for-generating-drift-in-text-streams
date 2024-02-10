import pandas as pd
import numpy as np

from random import choice
import nltk
from nltk.corpus import wordnet
import re

class AdjectiveSwap:
    def __init__(self, text, pos_tags_of_interest=['JJ', 'JJS']):
        self.text = text
        self.pos_tags_of_interest = pos_tags_of_interest

        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('omw-1.4', quiet=True)

    def get_pos_tag(self):
        lower_case = self.text.lower()
        tokens = nltk.word_tokenize(lower_case)
        tags = {word: tag for word, tag in nltk.pos_tag(tokens)}
        tags_words = {}
        words_tags = {}
        for w, t in tags.items():
            if t not in tags_words:
                tags_words[t] = [w]
            else:
                tags_words[t].append(w)

            if w not in words_tags:
                words_tags[w] = [t]
            else:
                words_tags[w].append(t)

        return tags_words, words_tags

    def get_antonyms(self, word):
        antonyms = []

        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                for antonym in lemma.antonyms():
                    antonyms.append(antonym.name())

        return list(set(antonyms))

    def get_generated_text(self):
        tags_words, words_tags = self.get_pos_tag()
        new_text = ""
        for i in self.text.split(" "):
            if len(i) < 2:
                new_text += f" {i}"
                continue

            if i in words_tags:
                if words_tags[i][0] in self.pos_tags_of_interest:
                    antonyms = self.get_antonyms(i)
                    if len(antonyms):
                        selected_antonym = choice(antonyms)
                        new_text += f" {selected_antonym}"
                    else:
                        new_text += f" {i}"
                else:
                    new_text += f" {i}"
        return new_text
    
def treat_text(text):
    text = str(text)
    text = re.sub('\s{2,}', ' ', text.lower().replace(".", " .").replace("\n", " ").replace(",", " ,").replace("!", " !").replace("?", " ?"))
    return text

def treat_text_for_adjective_swap(text):
    text = str(text)
    text = re.sub('\s{2,}', ' ', text.lower().replace(".", " .").replace("\n", " ").replace(",", " ,").replace("!", " !").replace("?", " ?"))
    ss = AdjectiveSwap(text)
    return ss.get_generated_text()

def sample_original_dataset(df, length, label_column, date_column, years_out=None, year_column=None):
    """
    Method for stratified sampling.
    """
    classes = df[label_column].unique()

    if years_out is not None:
        len_ds = len(df[~df[year_column].isin(years_out)])
    else:
        len_ds = len(df[label_column])

    prop = {e:0 for e in classes}
    amount = {e:0 for e in classes}
    for i in classes:
        if years_out is not None:
            prop[i] = len(df[(~df[year_column].isin(years_out))&(df[label_column] == i)]) / len_ds
        else:
            prop[i] = len(df[df[label_column] == i]) / len_ds

    final_size = length

    for e, (classe, _) in enumerate(sorted(prop.items(), key=lambda x: x[1], reverse=True)):
        amount[classe] = int(prop[classe] * final_size)
        if e == len(classes) - 1: #ultimo
            amount[classe] += final_size - sum(amount.values())

    if years_out is not None:
        dfs = [df[(~df[year_column].isin(years_out))&(df[label_column] == e)].sample(p) for e, p in amount.items()]
    else:
        dfs = [df[df[label_column] == e].sample(p) for e, p in amount.items()]

    final = pd.concat(dfs).sort_values(date_column, ascending=True).reset_index(drop=True)
    return final

def run(original_dataset: pd.DataFrame, data_scenarios, destination_folder, file_prefix, label_column='label', year_column='year',
        date_column='date', textual_column='review_treated', 
        map_shift={}, map_swap={},
        language_filter=None, language_column='', subset_number=10, subset_size=20000, drift_points_at_each=50000):
    for data_scenario in data_scenarios:
        if data_scenario == 1: # Class swap
            n_instances = subset_size
            drift_point = drift_points_at_each
            force_drift = True

            for i in range(subset_number):
                print(f"Iteration {i} scenario {data_scenario} - nodrift")

                df = original_dataset.copy()
                if language_filter is not None: df = df[df[language_column] == language_filter]

                df = sample_original_dataset(df, n_instances, label_column, date_column)[[textual_column, label_column, year_column]]
                df = df.astype({label_column:'int', year_column: 'int'})

                df.to_csv(f"{destination_folder}/{file_prefix}-nodrift-{i+1}-{data_scenario}.csv", index=False)
                
                print(f"Iteration {i} scenario {data_scenario} - withdrift")

                if force_drift:
                    df.loc[drift_point:,label_column] = df.loc[drift_point:,label_column].replace(map_swap)
                df.to_csv(f"{destination_folder}/{file_prefix}-withdrift-{i+1}-{data_scenario}.csv", index=False)

        elif data_scenario == 2: # Class shift
            n_instances = subset_size
            drift_point = drift_points_at_each
            force_drift = True

            for i in range(subset_number):
                print(f"Iteration {i} scenario {data_scenario} - nodrift")

                df = original_dataset.copy()
                if language_filter is not None: df = df[df[language_column] == language_filter]
                df = sample_original_dataset(df, n_instances, label_column, date_column)[[textual_column, label_column, year_column]]
                df = df.astype({label_column:'int', year_column: 'int'})

                df.to_csv(f"{destination_folder}/{file_prefix}-nodrift-{i+1}-{data_scenario}.csv", index=False)

                print(f"Iteration {i} scenario {data_scenario} - withdrift")
                if force_drift:
                    df.loc[drift_point:,label_column] = df.loc[drift_point:,label_column].replace(map_shift)
                    df.loc[drift_point*2:,label_column] = df.loc[drift_point*2:,label_column].replace(map_shift)
                    df.loc[drift_point*3:,label_column] = df.loc[drift_point*3:,label_column].replace(map_shift)
                df.to_csv(f"{destination_folder}/{file_prefix}-withdrift-{i+1}-{data_scenario}.csv", index=False)

        elif data_scenario == 3: # Time slice removal
            n_instances = subset_size

            for i in range(subset_number):
                print(f"Iteration {i} scenario {data_scenario}")

                df = original_dataset.copy()
                if language_filter is not None: df = df[df[language_column] == language_filter]
                years = sample_original_dataset(df, n_instances, label_column, date_column)[year_column].unique()[1:-1]

                choices = np.random.choice(years, 3, replace=False)
                with open(f'{destination_folder}/scenario-3-removed-years-{file_prefix}-{i+1}.txt', 'a') as f:
                    f.write(f"Iteration {i}: Years removed: {str(choices)}\n")

                df = sample_original_dataset(df, n_instances, label_column, date_column, choices, year_column)[[textual_column, label_column, year_column]]
                df = df.astype({label_column:'int', year_column: 'int'})

                df.to_csv(f"{destination_folder}/{file_prefix}-withdrift-{i+1}-{data_scenario}.csv", index=False)

        elif data_scenario == 4: # Adjective swap
            n_instances = subset_size
            drift_point = drift_points_at_each
            force_drift = True

            for i in range(subset_number):
                print(f"Iteration {i} scenario {data_scenario} - nodrift")
                df = pd.read_csv(f"{destination_folder}/{file_prefix}-nodrift-{i+1}-1.csv")
                df[textual_column] = df[textual_column].apply(treat_text)
                
                print(f"Iteration {i} scenario {data_scenario} - withdrift")

                if force_drift:
                    df.loc[drift_point:,textual_column] = df.loc[drift_point:,textual_column].apply(treat_text_for_adjective_swap)
                df.to_csv(f"{destination_folder}/{file_prefix}-withdrift-{i+1}-{data_scenario}.csv", index=False)

        elif data_scenario == 5: # Adjective swap (3x)
            n_instances = subset_size
            drift_point = drift_points_at_each
            force_drift = True

            for i in range(subset_number):
                print(f"Iteration {i} scenario {data_scenario} - nodrift")
                df = pd.read_csv(f"{destination_folder}/{file_prefix}-nodrift-{i+1}-1.csv")

                print(f"Iteration {i} scenario {data_scenario} - withdrift")
                if force_drift:
                    df.loc[drift_point:,textual_column] = df.loc[drift_point:,textual_column].apply(treat_text_for_adjective_swap)
                    df.loc[drift_point*2:,textual_column] = df.loc[drift_point*2:,textual_column].apply(treat_text_for_adjective_swap)
                    df.loc[drift_point*3:,textual_column] = df.loc[drift_point*3:,textual_column].apply(treat_text_for_adjective_swap)

                df.to_csv(f"{destination_folder}/{file_prefix}-withdrift-{i+1}-{data_scenario}.csv", index=False)

if __name__ == '__main__':
    """
    data_scenarios:
        - 1: Class swap
        - 2: Class shift
        - 3: Time slices removal
        - 4: Adjective swap
        - 5: Adjective swap (3x)
    """
    data_scenarios = [1, 2, 3, 4, 5]
    destination_folder = "generated_datasets"
    file_prefix = "airbnb"
    df = pd.read_csv("reviews-airbnb-enriched.csv")

    run(
        original_dataset=df,
        data_scenarios=data_scenarios,
        destination_folder=destination_folder,
        file_prefix=file_prefix, 
        map_swap={0:2, 2:0},
        map_shift={0:1, 1:2, 2:0},
        language_column='language',
        language_filter='en'
        )