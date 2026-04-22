# Experiments

This directory is for lightweight experiment notes.

Keep it simple. A single markdown file per idea or per day is enough.

Suggested format:

```md
# 2026-04-22 baseline

Command:
./run_simple.sh --train-steps 200

Change:
Default run to verify the setup.

Result:
- training loss:
- validation loss:
- runtime:

Next:
Try `--adam-head-lr 0.002 --adam-embed-lr 0.2`.
```

You do not need a heavy tracking system yet. The main goal is to keep enough notes that you can answer:
- what did I run?
- what changed?
- did it help?
- what should I try next?
