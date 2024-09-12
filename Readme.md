# Location Mention Recognition Using Flair and DeBERTa v3

This project focuses on recognizing and classifying location mentions in text using Named Entity Recognition (NER) with Flair's NLP framework. The model uses **DeBERTa v3 large embeddings** fine-tuned on the **OntoNotes 5 dataset**, with all location types unified under the `LOC` tag.

## Project Overview

- **Task**: Identify and classify location mentions (LOC) in text.
- **Framework**: Flair with **DeBERTa v3 large** transformer embeddings.
- **Dataset**: OntoNotes 5, with location types unified under `LOC`.
- **Tagging Scheme**: BIOES (Begin, Inside, Outside, End, Single).

## Preprocessing

1. **BIOES Tagging Scheme**: Convert the original BIO tagging scheme to BIOES to capture better entity boundaries.
   - **B-LOC**: Beginning of a multi-token location entity.
   - **I-LOC**: Inside a multi-token location entity.
   - **E-LOC**: End of a multi-token location entity.
   - **S-LOC**: Single-token location entity.
   - **O**: Outside of any entity.

2. **Unify Location Types**: All location types (e.g., GPE, LOC, FAC) in the dataset are unified under the `LOC` tag.

### Preprocessing Steps
Tagging Scheme Conversion: Convert the OntoNotes 5 dataset (or any other dataset you're using) from the existing tagging scheme (likely BIO) to BIOES (Begin, Inside, Outside, End, Single tagging).

* B (Begin) — Start of a multi-token entity.
* I (Inside) — Continuation of the entity.
* O (Outside) — Non-entity tokens.
* E (End) — Last token of a multi-token entity.
* S (Single) — Single-token entities.

`Unify Location Types`: In OntoNotes, multiple location types (e.g., GPE, LOC, FAC) exist. Convert all types to a unified tag LOC to treat all location mentions as a single category.



## Training Setup
`Embeddings`: Use DeBERTa v3 large embeddings, pre-trained, which are known for their state-of-the-art performance in NLP tasks.

`Training Parameters`:

`Epochs`: 3
`Learning Rate`: 5.0e-6
`Mini-batch Size`: 8
`Optimizer`: AdamW (Adam with weight decay)
`Embedding Reprojection`: To reduce the dimensionality of embeddings if needed, set reproject_embeddings=True in the Flair framework.

## Flair Framework Setup

        pip install flair
