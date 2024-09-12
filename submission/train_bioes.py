# train_bioes.py

import pandas as pd
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from typing import List

# Fonction pour appliquer le balisage BIOES
def bioes_tagging(text, location):
    words = text.split()
    tags = ['O'] * len(words)
    
    # Si la localisation est présente dans le texte
    if location:
        location_words = location.split()
        loc_len = len(location_words)
        
        for i in range(len(words)):
            # Vérifier si les mots correspondent à la localisation
            if words[i:i+loc_len] == location_words:
                if loc_len == 1:
                    tags[i] = 'S-LOC'  # Single
                else:
                    tags[i] = 'B-LOC'  # Begin
                    for j in range(1, loc_len - 1):
                        tags[i + j] = 'I-LOC'  # Inside
                    tags[i + loc_len - 1] = 'E-LOC'  # End
    return list(zip(words, tags))

# Charger le fichier de données d'entraînement
def prepare_training_data(train_csv):
    train_df = pd.read_csv(train_csv)
    with open('train_bioes.txt', 'w', encoding='utf-8') as f:
        for index, row in train_df.iterrows():
            text = row['text']
            location = row['location']
            if pd.notnull(text) and pd.notnull(location):
                tagged_words = bioes_tagging(text, location)
                for word, tag in tagged_words:
                    f.write(f"{word} {tag}\n")
                f.write("\n")  # Fin de la phrase

# Entraîner le modèle avec Flair
def train_flair_model():
    # Spécifier les colonnes
    columns = {0: 'text', 1: 'ner'}
    
    # Charger les données dans un corpus
    data_folder = './'
    corpus: Corpus = ColumnCorpus(data_folder, columns, train_file='train_bioes.txt')

    # Créer un dictionnaire de tags
    tag_type = 'ner'
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

    # Utiliser des embeddings empilés (Flair + Glove)
    embedding_types: List[TokenEmbeddings] = [
        WordEmbeddings('glove'),
        FlairEmbeddings('news-forward'),
        FlairEmbeddings('news-backward'),
    ]
    
    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    # Créer le modèle SequenceTagger
    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type,
                                            use_crf=True)

    # Créer un entraîneur de modèle
    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    # Entraîner le modèle
    trainer.train('ner-model',
                  learning_rate=0.1,
                  mini_batch_size=32,
                  max_epochs=10,
                  embeddings_storage_mode='gpu')

if __name__ == "__main__":
    # Préparer les données d'entraînement
    prepare_training_data('train.csv')

    # Entraîner le modèle
    train_flair_model()
