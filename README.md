# Transformer-Based Question Answering

A Natural Language Processing (NLP) project that builds an end-to-end **Question Answering (QA) system** using Hugging Face Transformers with both **DistilBERT** and **BERT** pre-trained models.

## Overview

This project demonstrates **Extractive Question Answering** - a task where the model reads a context paragraph and extracts the exact span of text that answers a given question. It uses two approaches:

1. **DistilBERT Pipeline** - High-level, plug-and-play approach using the Hugging Face `pipeline` API.
2. **BERT (Large, Whole Word Masking)** - Lower-level approach with manual tokenization and score-based span extraction for greater control.

Both models are fine-tuned on the **SQuAD (Stanford Question Answering Dataset)**.

## Models Used

| Model | Description |
|---|---|
| `distilbert-base-cased-distilled-squad` | Lightweight distilled version of BERT, fast & efficient |
| `bert-large-uncased-whole-word-masking-finetuned-squad` | Full BERT-Large, higher accuracy, fine-tuned on SQuAD |


## Usage

### Approach 1 - DistilBERT (Pipeline API)

```python
from transformers import pipeline

# Load pre-trained DistilBERT model
qa = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad",
    tokenizer="distilbert-base-cased"
)

context = """Your context paragraph goes here..."""

answer = qa(question="Your question here?", context=context)['answer']
print(answer)
```

### Approach 2 - BERT (Manual Tokenization)

```python
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad"
)
model = AutoModelForQuestionAnswering.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad",
    return_dict=False
)

context = """Your context paragraph goes here..."""
question = "Your question here?"

# Tokenize and encode
inputs = tokenizer.encode_plus(
    question, context,
    add_special_tokens=True,
    return_tensors="pt"
)

# Get answer span using scores
start_scores, end_scores = model(**inputs)
start_index = torch.argmax(start_scores)
end_index = torch.argmax(end_scores)

# Decode the answer
answer_tokens = inputs["input_ids"][0][start_index:end_index + 1]
answer = tokenizer.convert_tokens_to_string(
    tokenizer.convert_ids_to_tokens(answer_tokens, skip_special_tokens=True)
)
print(answer)
```

---

## Example Output

**Context used:**
> *It was a dark and stormy night. The old mansion on the hill loomed ominously, with its broken windows and overgrown vines. Inside, the air was musty and cold, and the only light came from flickering candles. As I made my way through the dusty halls, I couldn't shake the feeling that I was being watched. Suddenly, I heard a creaking sound behind me. I turned around to see a ghostly figure staring back at me with hollow eyes.*

**DistilBERT Results:**

| Question | Answer |
|---|---|
| What was the weather like on the night of the story? | `dark and stormy` |
| How was the ghost look like? | `hollow eyes` |
| How was the feeling while walking through the dusty halls? | `I couldn't shake the feeling that I was being watched` |

**BERT Result:**

| Question | Answer |
|---|---|
| How was the feeling while walking through the dusty halls? | `i was being watched` |



## References

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [DistilBERT Model Card](https://huggingface.co/distilbert-base-cased-distilled-squad)
- [BERT Large WW Masking Model Card](https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad)
- [SQuAD Dataset](https://rajpurkar.github.io/SQuAD-explorer/)
