# braid — team presentation script

A spoken script explaining how braid optimizes inference, for a non-technical audience
that knows sparkinfer. Every number is **measured** and published in the
[README](../README.md); the ~180 words-per-second ceiling is derived in
[ARCHITECTURE.md §1](ARCHITECTURE.md).

---

## The script

> A language model is basically a huge book of numbers, and to produce each word of an
> answer the graphics card has to read the entire book — so one user alone can never get
> more than about 180 words a second, no matter how clever the code is.
>
> The trick is sharing: if sixteen people are asking questions at once, you read the book
> once and answer all sixteen together.
>
> The catch is that this particular model has a "notebook" part that must be updated
> strictly one word at a time, which is why most engines give up and serve one person at
> a time.
>
> braid is built to not give up: we wrote custom GPU code so the notebook updates for
> many users at once, packed the numbers into a smaller format so far more users fit on
> one card, cut out thousands of tiny wasted steps the card was doing on every word, and
> added a scheduler that keeps everyone's requests flowing smoothly.
>
> The honest result: for a single user we're actually a bit slower than the popular
> alternative — and we publish that openly.
>
> But with sixteen users we're 26% faster, with sixty-four more than twice as fast, and
> at 128 users nearly three times as fast.
>
> Comparing full server against full server is where the gap is starkest: at sixty-four
> users we deliver almost six times the output, and at 128 their server has effectively
> collapsed — first answers take almost five minutes, while ours arrive in under a sixth
> of a second.
>
> And you already know how we keep ourselves honest, because it's the sparkinfer model:
> nothing counts unless a bot measures it on the real hardware, every proposed change
> gets benchmarked automatically, and a change that doesn't really help gets labeled as
> noise.
>
> Next we're making memory even smaller, making repeat conversations start faster, and
> moving up to a much bigger model.

## Presenter notes

- **"Why be slower for one user?"** Single-user speed is a solved, crowded race; serving
  many users at once is the open one. Capacity scales with users served, not with one
  user's speed — and we publish the losing single-user row unchanged because hiding it
  would make the whole curve untrustworthy.
- **Number sources**, all measured: the decode curve (−22.0% at B=1, +26.5% at B=16,
  +118.8% at B=64, +170.4% at B=128) and the server head-to-head (+72% at c=8 → +473% at
  c=64; TTFT 287 s vs 158 ms at c=128) are the README's published tables. Quote them
  exactly rather than rounding further.
- **Do not say** the competition "can't fit 128 users on the card" — llama.cpp runs
  B=128 fine in-process (2,662.8 tok/s). What fails at 128 is its *server* (TTFT ~5 min);
  it's braid's own non-FP8 configurations that don't fit, which is the fp16-state story.
- If wall-of-memory percentages come up, use the README's published pair: llama.cpp at
  88% of the wall at batch 1, 12% at batch 128. The 35%/13% figures floating around in
  THESIS are the older 4B measurements.
