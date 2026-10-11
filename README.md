# Model development

By **[Chase Dovey](https://cdovey.dev/)** · [Average Joes Lab](https://github.com/averagejoeslab)

Model development, end to end: build a language model's architecture from its primitives, then train it through the stages a lab uses (pre-training, mid-training and post-training) until it works as an agent. It all runs on one computer, with no GPU.

Every number in this README comes from a real run, and the run is in [`runs/`](./runs/).

```
model.py    the model: its seven primitives, checked against Qwen3-0.6B
train.py    its training, stage by stage: pre → mid → instruction-tuning → RL
harness.py  the smallest agent around our model: where post-training happens, and where it's checked
tasks.py    the tasks the agent trains on, each with a check of what it left behind
data/       what training reads          models/  what training writes (not committed)
runs/       what each stage printed
```

## The model

A model predicts. Tokens go in, and it outputs the most likely next token, which is added to the input. It repeats until it outputs an end token.

```
            ┌──────────── append ◄────────────┐
            ▼                                 │
TokensIn ─► Model ─► most likely next token ──┤
                                              │
                                    is it the end? ── yes ─► TokensOut
```

**TokensOut = Model(TokensIn)**

### Its primitives

Every model of this kind is built from the same seven primitives:

| Primitive | What it does |
|---|---|
| **Tokenizer** | turns text into tokens (numbers the model can work on) and back |
| **Embedding** | turns each token into a list of numbers that stands for its meaning |
| **Position** | marks where each token is |
| **Attention** | lets each token take in the ones before it |
| **Block** | attention, then a small network; stacked many times |
| **Output head** | gives a score to every possible next token |
| **Generation** | picks the next token, adds it, and goes again |

Model families differ in how they implement each one. We build Qwen3's version, so we can load Qwen's trained numbers into our code and check it against theirs.

In a trained model, they work together like this. These are the real numbers from [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B-Base):

```
"The capital of France is"
        │  tokenizer
        ▼
[785, 6722, 315, 9625, 374]                      5 tokens
        │  embedding + position
        ▼
5 × 1,024 numbers                                each token's meaning, and where it is
        │  28 blocks: attention (look back) + network (think about it)
        ▼
5 × 1,024 numbers                                each token, now in context
        │  output head
        ▼
5 × 151,936 scores                               for every possible next token, at every position
        │  generation: take the last position's highest score
        ▼
" Paris"                                         added to the input, and round again
```

### Building them

Each primitive gets one line on what it does. The code from [`model.py`](./model.py) shows how.

**Tokenizer: turns text into tokens, and back.**

```python
def learn(cls, text, size, special=()):             # byte-pair encoding: start from single bytes...
    while 256 + len(merges) + len(special) < size:
        pairs = collections.Counter()
        for w, n in words.items():
            for pair in zip(w, w[1:]):
                pairs[pair] += n                    # ...count every neighbouring pair...
        best = max(pairs, key=pairs.get)
        merges.append(best)                         # ...and make the commonest one a new token
        words = {cls.merge(w, best): n for w, n in words.items()}
```

To encode a word, it's split into bytes and the learned merges are applied, commonest first. Common words end up as one token, rare ones as several.

**Embedding: gives each token a meaning.**

```python
self.embed_tokens = nn.Embedding(vocab, dim)     # a row of 1,024 numbers per token; training puts related tokens close
x = self.embed_tokens(ids)
```

**Position: tells each token where it is.**

```python
frequencies = 1.0 / theta ** (torch.arange(0, dim, 2).float() / dim)
angles = torch.outer(torch.arange(start, start + length).float(), frequencies)   # an angle that grows along the input
q, k = rotate(q, cos, sin), rotate(k, cos, sin)  # turn each query and key by its angle: nearby tokens line up
```

**Attention: each token gathers what it needs from the tokens before it.**

```python
q = self.q_norm(self.q_proj(x).view(batch, length, self.heads, self.head_dim)).transpose(1, 2)      # what each token looks for
k = self.k_norm(self.k_proj(x).view(batch, length, self.kv_heads, self.head_dim)).transpose(1, 2)   # what each token offers
v = self.v_proj(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)                # what each token carries
scores = q @ k.transpose(-2, -1) / self.head_dim ** 0.5          # how well each query matches each key
weights = scores.masked_fill(ahead, float("-inf")).softmax(dim=-1)   # never look ahead; share out attention by match
mixed = weights @ v                              # take that share of each value
```

It does this 16 times side by side. Each copy is a *head*, free to look for something different.

**Block: looks back, then thinks about it.**

```python
x = x + self.self_attn(self.input_layernorm(x), cos, sin, cache)               # look back (attention)
h = self.post_attention_layernorm(x)                                            # keep the numbers a steady size
gate = self.gate_proj(h)
return x + self.down_proj(gate * torch.sigmoid(gate) * self.up_proj(h))        # think about it: a gated network
```

Each step adds to what came in, so nothing is lost on the way through. Qwen3-0.6B stacks 28 blocks.

**Output head: scores every possible next token.**

```python
return x @ self.embed_tokens.weight.T            # compare with every token's meaning: 151,936 scores
```

**Generation: picks the next token and goes again.**

```python
scores = scores / temperature                    # sharpen or flatten
scores = scores.masked_fill(scores < scores.topk(top_k).values[:, -1:], float("-inf"))   # keep the k likeliest
ordered, order = F.softmax(scores, dim=-1).sort(descending=True)
ordered[ordered.cumsum(-1) - ordered >= top_p] = 0                    # and only as many as make up top_p
return order.gather(-1, torch.multinomial(ordered, 1))           # draw one
```

```python
scores = self.model(next_id, caches, start=ids.shape[1] + len(out) - 1)[:, -1]   # add it, go again; the cache keeps the rest
```

That's the whole model. A model also ships with its tokenizer, a *chat template* (the way it lays out a conversation) and settings for generating. Miss one and it breaks quietly, so each is loaded from its own file. Load Qwen's trained numbers into our code and check it against Qwen's own implementation (`uv run model.py`, [`runs/model/check.txt`](./runs/model/check.txt)):

```
tokenizer: the same tokens as Qwen's: True
chat template: the same text as Qwen's: True
model: largest difference in its scores 1.4e-04; the same next token at all 198 positions: True
'The capital of France is' → ' Paris. The capital of Germany is Berlin. The capital of'
```

## Training it

Built, the model is untrained: its numbers are random, and it outputs noise. Training fills them in, and it's one idea, repeated. Give the model tokens and measure how surprised it was by each real next token (the *loss*). Then nudge every number so it's less surprised next time:

```python
loss = F.cross_entropy(model(inputs).flatten(0, 1), targets.flatten())   # how surprised was it?
optimizer.zero_grad(); loss.backward(); optimizer.step()                # nudge every number to be less so
```

Labs split training into stages. Each stage has its own goal, so it uses different data and a different idea of what's right:

| Stage | Goal | Data | What counts as right |
|---|---|---|---|
| **Pre-training** | know language and the world | everything: web, books, code; trillions of tokens | the next token, everywhere |
| **Mid-training** | be good at what matters | chosen: maths, code, a domain | the next token, everywhere |
| **Post-training: instruction-tuning** | behave: take turns, use tools, stop | example conversations | the assistant's tokens |
| **Post-training: RL** | succeed, reliably | its own attempts, graded | whatever earns reward |

Pre- and mid-training put knowledge in. Post-training can't add much knowledge; it shapes behaviour, and the model becomes whatever the grading rewards. Each stage needs the one before.

Every stage is checked before and after, on held-out data from its own source that it never trained on. The question is always: **did it train what it should?**

### Pre-training

**Goal:** language and knowledge. We pre-train our model from random numbers. It's the same code with 4 blocks instead of 28 (3.7 million numbers instead of 600 million), trained for 20 minutes on about a million characters of Shakespeare's plays. Its tokenizer learns 2,048 tokens from the same text. The last tenth is held out, never trained on, to measure it fairly (`uv run train.py pre`):

| | Held-out loss | Continuing `ROMEO:` |
|---|---|---|
| **Untrained** | ⟨pending⟩ | ⟨pending⟩ |
| **After 20 minutes** | ⟨pending⟩ | ⟨pending⟩ |

The same code, trained on about 36 trillion tokens for months on a cluster of GPUs, writes far better. That's over 100 million times more data than ours, and it's the one stage a single computer can't afford. So from here on we load a model that's already pre-trained: **Qwen3-0.6B-Base**, after Qwen's pre- and mid-training ([their report](https://arxiv.org/abs/2505.09388)) and before any post-training. Our code is the same at every scale, so its numbers load straight in:

```
ROMEO:
⟨pending⟩
```

It's fluent, but nothing has taught it how to behave.

### Mid-training

**Goal:** know the domain. Mid-training is the same task as pre-training, predicting every next token, on tokens chosen for the job. Our model will act through bash, so the data is [tldr-pages](https://github.com/tldr-pages/tldr) (CC BY 4.0): about 6,800 short pages on shell commands, each a plain-English line and the command for it:

```
- [c]reate a g[z]ipped archive and write it to a [f]ile:

`tar czf {{path/to/target.tar.gz}} {{path/to/file1 path/to/file2 ...}}`
```

200 pages are held out. Training runs a fixed 60 steps, about an eighth of the rest (`uv run train.py mid`). There are two checks, before and after:

- the loss on the held-out pages
- a shell check on 100 of them

A base model is asked by example, so each page is shown with its other examples first, then its last request. A command passes if it uses the right program and includes every word of the page's command, in any order, with any value for a `{{placeholder}}`.

| | Held-out loss | Shell check |
|---|---|---|
| **Before: Qwen3-0.6B-Base** | ⟨pending⟩ | ⟨pending⟩ |
| **After 60 steps** | ⟨pending⟩ | ⟨pending⟩ |

### Post-training

Base continues text. It was never taught to take turns, or even to end one. Post-training teaches it, in the chat format of Qwen3-0.6B (Qwen's own chat model).

**Our model in a harness.** We post-train an agent, so the model goes into a harness: the code around it that turns its tokens into actions, and the results back into tokens. [`harness.py`](./harness.py) is the smallest one, five primitives in about 70 lines:

| Primitive | What it does |
|---|---|
| **Input** | captures what comes in: a request |
| **Context** | lays out the instructions and the conversation in the model's chat template |
| **Model interface** | asks our model for a reply |
| **Output** | shows its words, and runs each `<tool_call>` it writes in bash |
| **Control flow** | if a tool ran, its result goes back and the model goes again; if not, it hands back |

```python
def act(conversation, steps=None):                       # the inner loop: go again while a tool ran; in training, at most `steps`
    while steps is None or steps > 0:
        response = request_response(assemble_context(conversation))
        conversation.append({"role": "assistant", "content": response})
        tool_results = handle_output(response)
        if not tool_results:
            return                                       # done: hand back to the person
        conversation += tool_results                     # a tool ran: go again
        steps = steps and steps - 1
```

**The tasks.** [`tasks.py`](./tasks.py) has 26 kinds of work in small folders: a log, a CSV, a config, notes, failing tests. Examples:

- *Save the last 3 lines of server.log to recent.txt.*
- *What port does config.json set? Just tell me, don't touch it.*
- *The tests are failing. Can you fix them?*

Each task is a fresh folder, a request, a reference solution, and a check of what the agent left behind, not what it said. Training uses one set of folders. The **held-out tasks** are 40 others (every kind, in folders training never sees), run in the harness, and they're how post-training is checked.

The mid-trained model in the harness, on the held-out tasks (`uv run train.py score models/midtrain.pt`): **⟨pending⟩/40**. It knows the shell, but it doesn't know it's in a conversation.

#### Instruction-tuning

**Goal:** take turns, call the tool in the harness's format, stop. The model sees example conversations, and it learns from the assistant's tokens only. Four kinds, 2,100 in all:

| Kind | Data | How many |
|---|---|---|
| Talk | [smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) (Apache 2.0), nothing about the shell | 200 |
| Ask for a command | [NL2Bash](https://github.com/TellinaTool/nl2bash) | 200 |
| Reason | [GSM8K](https://huggingface.co/datasets/openai/gsm8k) (MIT): worked steps, as thinking | 200 |
| Work as the agent | our training tasks: every command really run, kept only if it passes its check | 1,500 |

A session looks at the folder, acts, checks its work, then answers from what it saw:

```
<|im_start|>user
Set debug to true in config.json, and leave the rest alone.<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "bash", "arguments": {"command": "python3 -c \"import json; c = json.load(open('config.json')); …\""}}
</tool_call><|im_end|>                                    ← learned
<|im_start|>user
<tool_response>
(no output)
</tool_response><|im_end|>                                ← read, not learned
<|im_start|>assistant
<tool_call>
{"name": "bash", "arguments": {"command": "cat config.json"}}
</tool_call><|im_end|>                                    ← learned: check before answering
<|im_start|>user
<tool_response>
{ "name": "backup", "port": 8346, "debug": true }
</tool_response><|im_end|>
<|im_start|>assistant
Done: config.json now has debug set to true: …<|im_end|>   ← learned: answer from what it saw
```

(shortened)

Nothing it trains on may resemble what it's checked on. Public examples that share 40% or more of their words with a held-out task are left out.

It trains every number, the embedding table too, since Base has barely learned the token that ends a turn. It makes one pass through the data, because a second starts memorising (`uv run train.py instruct`).

Held-out loss on 48 conversations it never trained on: **⟨pending⟩ → ⟨pending⟩**. In the harness, on the held-out tasks: **⟨pending⟩/40**.

⟨pending: the same requests before and after, and the ways it still fails⟩

#### Reinforcement learning

**Goal:** succeed reliably. The model tries each kind of training task 8 times in the harness. Each try is graded by its task's check, and the tries that passed are made more likely. If all 8 pass or all 8 fail, there's nothing to learn from. So RL can only make more likely what the model already does some of the time (`uv run train.py rl`).

Imitation and RL are the same function, with different weights. Imitation weights every example 1. RL weights a try that passed 1 and a try that failed 0:

```python
def update(model, optimizer, texts, weights, most=1024):  # one step: make each text more likely, in proportion to its weight
    optimizer.zero_grad()
    for text, weight in zip(texts, weights):             # one at a time, to fit in memory
        (-weight * logprob(model, text, most) / len(texts)).backward()
    optimizer.step()
```

A lab's RL also makes failed tries *less* likely. It stops the model drifting with a penalty for straying from a reference copy, which needs memory for a second model. Without that anchor, a model this small drifts off its format when pushed away from its own words. So we keep the wins only, at a tenth of instruction-tuning's step size.

Held-out tasks: **⟨pending⟩/40 → ⟨pending⟩/40**. Share of tries that passed in training, round by round ([`runs/rl.txt`](./runs/rl.txt)):

| Round | Tries passed |
|---|---|
| 1 | ⟨pending⟩ |
| 2 | ⟨pending⟩ |
| 3 | ⟨pending⟩ |

⟨pending: what improved, and what never passed a try⟩

## Stage by stage

Each stage, on its own check:

| Stage | Its check | Before | After |
|---|---|---|---|
| Pre-training (ours) | held-out loss on Shakespeare | ⟨pending⟩ | ⟨pending⟩ |
| Mid-training | held-out loss on shell pages; the shell check | ⟨pending⟩ | ⟨pending⟩ |
| Instruction-tuning | held-out loss on conversations | ⟨pending⟩ | ⟨pending⟩ |
| Reinforcement learning | held-out tasks in the harness | ⟨pending⟩ | ⟨pending⟩ |

The same 40 held-out tasks in the harness, where only the model changes:

| Model in the harness | Held-out tasks |
|---|---|
| after mid-training | ⟨pending⟩/40 |
| after instruction-tuning | ⟨pending⟩/40 |
| after reinforcement learning | ⟨pending⟩/40 |
| Qwen3-0.6B: Qwen's own post-training of the same base | ⟨pending⟩/40 |

⟨pending: what fell short, its likely cause, and what would fix it⟩

## Run it

You need [uv](https://docs.astral.sh/uv/). There's no GPU and no API key: everything runs on a CPU, in about ⟨pending⟩ hours on four cores. Each command writes what it printed to `runs/`.

```bash
uv run model.py                          # check the model against Qwen's own
uv run train.py pre                      # pre-training: our model, from random numbers, on Shakespeare
uv run train.py mid                      # mid-training: Qwen3-0.6B-Base, on shell pages
uv run train.py score models/midtrain.pt # the agent before post-training, on the held-out tasks
uv run train.py instruct                 # post-training: instruction-tuning
uv run train.py rl                       # post-training: reinforcement learning, in the harness
uv run train.py score models/rl.pt
uv run harness.py models/rl.pt           # talk to it
```

> [!WARNING]
> In RL and scoring, the harness runs the commands our model writes, without asking. Run it somewhere you can afford to lose, as a user that can't harm the rest of your machine.
