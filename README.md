# LM-Variation

This project investigates the variation in how neural language models learn abstract
properties of lexical items. In particular, we work with BERT, GPT-2, and Transformer-XL,
exploring their performances on grammatical tasks and what factors contribute to them.

## Repository overview
| Directory | Description |
|---|---|
| `compute` | Evaluation of language model performance on grammatical tasks |
| `one_shot` | Finetuning scripts, data, and qualitative evaluation for one/few-shot learning with BERT and GPT-2 |
| `plotting` | Scripts for producing plots from model performance results |
| `sentence_generation` | Generated token and sentence lists of various templates for model evaluation |
| `transformer_evals` | Results from evaluating the transformers on agreement and anaphora tasks |

## Dependencies

```bash
# (Python 3.x)
pip install -r requirements.txt
```
