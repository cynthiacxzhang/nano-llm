# nano-llm

A GPT implementation from scratch in ~200 lines of Python.

Built to understand — not just run — the transformer architecture: what each component does,
why the design choices exist, and what happens at the hardware level.

Trained on UN General Assembly resolutions.

## structure

```
model.py      — transformer architecture, annotated
train.py      — training loop with explicit memory calculations
generate.py   — sampling / inference
data/         — corpus and preprocessing
writeup.md    — companion essay
```

## running

```bash
python train.py
python generate.py --prompt "The General Assembly,"
```

## writeup

See [`writeup.md`](writeup.md) for the companion essay covering architecture decisions,
hardware constraints, and what the model actually learned.
