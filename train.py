# /// script
# dependencies = ["torch", "numpy", "regex", "safetensors", "huggingface_hub", "jinja2", "datasets"]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Training the model, one stage at a time, each starting from the last: uv run train.py pre | mid | instruct | rl | score <weights>"""
import sys, os, re, io, glob, json, time, random, shutil, functools, contextlib, subprocess, urllib.request
from datasets import load_dataset
import torch, torch.nn.functional as F
from model import Model, Tokenizer, Release, load_real, generate
import tasks, harness

ROOT = os.path.dirname(os.path.abspath(__file__))        # data/ comes in, models/ come out, runs/ is the record
os.chdir(ROOT)
LOG = None                                               # runs/<stage>.txt: what each stage printed, kept
CHAT, BASE = "Qwen/Qwen3-0.6B", "Qwen/Qwen3-0.6B-Base"   # Qwen's chat format and tokenizer; Base's pre-trained numbers


def log(*words):
    line = " ".join(str(w) for w in words)
    print(line, flush=True)
    if LOG:
        open(LOG, "a").write(line + "\n")


def logging_to(stage):                                   # a fresh record for this stage's run; a resumed run adds to it
    global LOG
    LOG = f"runs/{stage}.txt"
    if not os.path.exists(f"models/{stage}.progress"):
        open(LOG, "w").close()


# ── 1. pre-training: predict the next token of raw text ───────────────────────

PROMPT = "ROMEO:\n"


def pretrain_text():                                     # Shakespeare: about a million characters of plays
    if not os.path.exists("data/shakespeare.txt"):
        open("data/shakespeare.txt", "w").write(urllib.request.urlopen("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt").read().decode())
    return open("data/shakespeare.txt").read()


def pretrain(minutes=20):
    logging_to("pretrain")
    text = pretrain_text()
    tokenizer = Tokenizer.learn(text, 2048)              # our own tokenizer, learned from the same text
    data = torch.tensor(tokenizer.encode(text))
    cut = len(data) * 9 // 10
    data, held_out = data[:cut], data[cut:]              # the last tenth: never trained on, to measure it fairly
    model = Model(vocab=2048, dim=256, layers=4, heads=4, kv_heads=2, head_dim=64, hidden=768)   # the same design, much smaller
    log(f"text: {len(data):,} tokens of {len(tokenizer.ids):,} kinds; model: {sum(p.numel() for p in model.parameters()):,} numbers, all random")
    speak = lambda: tokenizer.decode(sample(model, tokenizer.encode(PROMPT), 60))
    log("before:", repr(PROMPT + speak()), f"held-out loss {held_out_loss(model, held_out):.2f}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    start, step = time.time(), 0
    while time.time() - start < minutes * 60:
        step += 1
        starts = torch.randint(len(data) - 257, (16,))
        inputs = torch.stack([data[i:i + 256] for i in starts])
        targets = torch.stack([data[i + 1:i + 257] for i in starts])   # the same text, one token on: the next one
        loss = F.cross_entropy(model(inputs).flatten(0, 1), targets.flatten())   # how surprised was it?
        optimizer.zero_grad(); loss.backward(); optimizer.step()                # nudge every number to be less so
        if step % 100 == 0:
            log(f"step {step}: loss {loss.item():.2f}")
    log(f"after {step} steps, {minutes} minutes:", repr(PROMPT + speak()), f"held-out loss {held_out_loss(model, held_out):.2f}")
    real = load_real()                                   # now Qwen's: its numbers, and its tokenizer, in the same code
    log("Qwen3-0.6B-Base, trained on about 36 trillion tokens:", repr(PROMPT + generate(real, PROMPT, most=60)))


@torch.no_grad()
def held_out_loss(model, data):                          # the fixed score: how well it predicts text it never saw
    chunks = data[:len(data) // 257 * 257].view(-1, 257)
    return F.cross_entropy(model(chunks[:, :-1]).flatten(0, 1), chunks[:, 1:].flatten()).item()


@torch.no_grad()
def sample(model, ids, most):
    ids = torch.tensor([ids])
    for _ in range(most):
        next_id = torch.multinomial(F.softmax(model(ids[:, -256:])[:, -1] / 0.8, dim=-1), 1)
        ids = torch.cat([ids, next_id], dim=1)
    return ids[0, -most:].tolist()


# ── 2. mid-training: the same task, on the agent's domain ─────────────────────

def ours(weights):                                       # our model: Qwen's chat format and tokenizer, with these numbers
    return Release(CHAT, weights)


@functools.cache
def held_out_requests():                                 # the requests of the held-out tasks: what training data must not resemble
    requests = []
    for n in range(40):
        where, ask, *_ = tasks.training_task(10_000 + n, tasks.TRAINING[n % len(tasks.TRAINING)])
        shutil.rmtree(where)
        requests.append((ask, words(ask)))
    return requests


def words(request):                                      # what a request asks for: its words, without the ones every request has
    return set(re.findall(r"[a-z]+", request.lower())) - set("a an the and or of in into to it its this that is are be with from for on at all every each".split())


def near(request):                                       # decontamination: shares 40% or more of its words with a held-out request
    mine = words(request)
    return max(len(mine & theirs) / len(mine | theirs or {""}) for _, theirs in held_out_requests()) >= 0.4


def shell_pages():                                       # tldr-pages (CC BY 4.0): short pages on shell commands, with examples
    where = "data/tldr"
    if not os.path.isdir(where):
        subprocess.run(f"git clone -q --depth 1 --filter=blob:none --sparse https://github.com/tldr-pages/tldr.git {where} && "
                       f"git -C {where} sparse-checkout set pages/common pages/linux", shell=True, check=True)
    return [open(path).read() for path in sorted(glob.glob(f"{where}/pages/*/*.md"))]


def examples(page):                                      # a page's examples: what it does, in plain English, and the command
    return [(re.sub(r"[\[\]]", "", what), command) for what, command in re.findall(r"^- (.+?):\n\n`(.+?)`$", page, re.M)]


def asked(page):                                         # the page up to its last example's command, as the page itself shows it
    head, command = page[:page.rindex("`", 0, page.rindex("`"))], examples(page)[-1][1]
    return head + "`", command


def parts(command):                                      # a command's words: a {{placeholder}} stays whole, spaces and all
    return re.findall(r"(?:\{\{.+?\}\}|[^\s{])+", command.removeprefix("sudo "))


def pattern(word):                                       # a reference word as a pattern: any value for a {{placeholder}}; either form of {{[-a|--all]}}
    return "".join("(" + "|".join(map(re.escape, part[3:-3].split("|") + [part])) + ")" if part.startswith("{{[") else
                   ".+?" if part.startswith("{{") else re.escape(part) for part in re.split(r"(\{\{.+?\}\})", word))


def forms(word):                                         # the ways a word can be read: a {{[-a|--all]}} it wrote is either form too
    return [word] + (word[3:-3].split("|") if re.fullmatch(r"\{\{\[.+\]\}\}", word) else [])


def fits(said, command):                                 # the right command: the same first word, and every word of the reference, in any order
    said, wanted = parts(said), parts(command)
    return bool(said) and said[0] == wanted[0] and all(any(re.fullmatch(pattern(w), f) for s in said for f in forms(s)) for w in wanted[1:])


@torch.no_grad()
def shell_check(model, pages):                           # asked as a base model is asked, by example: the page's own examples, then its last request
    right = sum(fits(model.generate(prompt, most=48, temperature=0).split("`")[0], command) for prompt, command in map(asked, pages))
    return f"{right}/{len(pages)}"


def chunks(model, texts, length=512):                    # the pages as one stream of tokens, cut into equal pieces
    ids = [i for t in texts for i in model.encode(t + "\n\n")]
    return torch.tensor(ids[:len(ids) // (length + 1) * (length + 1)]).view(-1, length + 1)


@torch.no_grad()
def chunk_loss(model, data):
    return sum(F.cross_entropy(model.model(c[None, :-1])[0], c[1:]).item() for c in data) / len(data)


def prepared(name, make):                                # data that's slow to make, made once and kept, so a restart reuses it
    path = f"data/{name}.pt"
    if not os.path.exists(path):
        torch.save(make(), path)
    return torch.load(path)


def mid_data(model):
    every = shell_pages()
    pages = [p for p in every if not any(near(what) for what, _ in examples(p))]   # decontaminated
    random.shuffle(pages)
    data, held_out = chunks(model, pages[200:]), chunks(model, pages[:200])[:32]   # 200 pages it never trains on
    tests = [p for p in pages[:200] if len(examples(p)) > 1 and "alias of" not in p][:100]   # the shell check: pages with examples to go by
    log(f"{len(pages):,} pages ({len(every) - len(pages)} left out, too near a held-out task), {data.numel():,} tokens")
    return data, held_out, tests


def midtrain(steps=60):                                  # a fixed number of steps, so a rerun does the same: about an eighth of the pages
    logging_to("midtrain")
    model = ours(BASE)
    data, held_out, tests = prepared("midtrain", lambda: mid_data(model))
    done = resume(model, "midtrain")
    if not done:
        log(f"held-out loss before: {chunk_loss(model, held_out):.3f}; shell check before: {shell_check(model, tests)} right commands")
        save(model.model, "midtrain", step=0)            # the checks before are done: a restart goes straight to training
    optimizer = trainable(model.model, lr=1e-5)
    for step in range(done.get("step", 0) + 1, steps + 1):
        optimizer.zero_grad()
        for piece in data[4 * step - 4:4 * step]:        # four pieces a step, one at a time to fit in memory
            loss = F.cross_entropy(model.model(piece[None, :-1])[0], piece[1:])
            (loss / 4).backward()
        optimizer.step()
        if step % 5 == 0:
            save(model.model, "midtrain", step=step)
        if step % 20 == 0:
            log(f"step {step}: loss {loss.item():.3f}")
    log(f"after {steps} steps; held-out loss after: {chunk_loss(model, held_out):.3f}; shell check after: {shell_check(model, tests)} right commands")
    finish(model, "midtrain")


# ── 3. post-training, part one: instruction-tuning ────────────────────────────

def gsm8k(split):                                        # grade-school maths: a question, worked steps, a number
    return list(load_dataset("openai/gsm8k", "main", split=split).shuffle(seed=0))


def worked(answer):                                      # its worked answer: the steps as thinking, then the answer
    steps, number = answer.split("####")
    return "<think>\n" + re.sub(r"<<.*?>>", "", steps).strip() + f"\n</think>\n\nThe answer is {number.strip()}."


def instruction_data(model):                             # four kinds of conversation, in the model's own format, none near a held-out task
    smoltalk = load_dataset("HuggingFaceTB/smol-smoltalk", split="train").shuffle(seed=0)
    shell = re.compile(r"bash|shell|terminal|command line|linux|unix|script", re.I)   # talk stays away from the tool: the model acts there, not explains
    chats = [m for m in (row["messages"] for row in smoltalk.select(range(20000)))
             if sum(len(x["content"]) for x in m) < 1500 and not near(m[0]["content"]) and not shell.search(" ".join(x["content"] for x in m))][:200]
    nl2bash = list(zip(*(urllib.request.urlopen(f"https://raw.githubusercontent.com/TellinaTool/nl2bash/master/data/bash/all.{part}").read().decode().splitlines() for part in ("nl", "cm"))))
    data = [model.chat(chat, prompt=False) for chat in chats]                                                        # talk
    data += [model.chat([tasks.system(), {"role": "user", "content": ask}, tasks.call(command)], tools=[tasks.BASH], prompt=False)
             for ask, command in random.sample([row for row in nl2bash if not near(row[0])], 200)]                     # ask for a command
    data += [model.chat([{"role": "user", "content": row["question"]}, {"role": "assistant", "content": worked(row["answer"])}], prompt=False)
             for row in gsm8k("train")[:200]]                                                                         # reason
    data += [model.chat(tasks.tool_session(), tools=[tasks.BASH], prompt=False) for _ in range(1500)]               # work as the agent: verified sessions
    random.shuffle(data)
    return data


def tokens_and_mask(model, text, most=1024):              # learn only what the assistant says, and the end of its turn
    ids, learn = [], []
    for n, piece in enumerate(re.split(r"(?<=<\|im_start\|>assistant\n)(.*?<\|im_end\|>)", text, flags=re.S)):
        piece_ids = model.encode(piece)
        ids += piece_ids
        learn += [n % 2 == 1] * len(piece_ids)
    return ids[:most], learn[:most]


def logprob(model, text, most=1024):                      # how likely the model finds what the assistant said, per token
    ids, learn = tokens_and_mask(model, text, most)
    ids, keep = torch.tensor([ids]), torch.tensor([learn[1:]])
    return -F.cross_entropy(model.model(ids[:, :-1], keep=keep), ids[:, 1:][keep])


def update(model, optimizer, texts, weights, most=1024):  # one step: make each text more likely, in proportion to its weight
    optimizer.zero_grad()
    for text, weight in zip(texts, weights):             # one at a time, to fit in memory
        (-weight * logprob(model, text, most) / len(texts)).backward()
    optimizer.step()


@torch.no_grad()
def held_out(model, texts):
    return -sum(logprob(model, t).item() for t in texts) / len(texts)


def trainable(model, lr):                                # every number, the vocabulary too: Base barely knows <|im_end|>
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)


def save(model, name, **progress):                       # the numbers, and how far it got, so an interruption costs little
    torch.save({k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}, f"models/{name}.pt")
    json.dump(progress, open(f"models/{name}.progress", "w"))


def resume(model, name):                                 # carry on from the last save, if there is one (the optimizer starts afresh)
    if not os.path.exists(f"models/{name}.progress"):
        return {}
    model.model.load_state_dict({k: v.float() for k, v in torch.load(f"models/{name}.pt").items()})
    progress = json.load(open(f"models/{name}.progress"))
    log(f"carrying on from {progress}")
    return progress


def finish(model, name):
    save(model.model, name)
    os.remove(f"models/{name}.progress")


def show(model):                                         # real replies: talk, a tool call, and reasoning
    for messages, tools_, thinking in [([{"role": "user", "content": "How do I make a cup of tea?"}], None, False),
                                       ([tasks.system(), {"role": "user", "content": "Count the lines in notes.txt."}], [tasks.BASH], False),
                                       ([{"role": "user", "content": gsm8k("test")[0]["question"]}], None, True)]:
        log(f"  > {messages[-1]['content']}\n  < " + model.generate(model.chat(messages, tools_, thinking), most=300, temperature=0).replace("\n", "\n    "))


def instruct():
    logging_to("instruct")
    model = ours("models/midtrain.pt")
    data = prepared("instruct", lambda: instruction_data(model))
    test, data = data[:48], data[48:]                    # held out: the fixed score for this stage
    done = resume(model, "instruct")
    if not done:
        log(f"{len(data)} conversations; held-out loss before: {held_out(model, test):.3f}")
        log("before:"); show(model)
        save(model.model, "instruct", step=0, minutes=0)  # the checks before are done: a restart goes straight to training
    optimizer, start, earlier = trainable(model.model, lr=1e-5), time.time(), done.get("minutes", 0)
    for step in range(done.get("step", 0) + 1, len(data) // 8 + 1):   # one pass: a second starts memorising
        update(model, optimizer, data[8 * step - 8:8 * step], [1.0] * 8)    # every example counts the same: imitate it
        minutes = earlier + (time.time() - start) / 60
        if step % 10 == 0:
            save(model.model, "instruct", step=step, minutes=minutes)
        if step % 25 == 0:
            log(f"step {step}: {minutes:.0f} minutes")
    log(f"after {step} steps, {minutes:.0f} minutes; held-out loss after: {held_out(model, test):.3f}")
    log("after:"); show(model)
    finish(model, "instruct")


# ── 4. post-training, part two: reinforcement learning, in the harness ──────

def attempt(kind, seed, steps=10):                       # one try at a task, in the harness, in a fresh folder: what it said and did, and did it pass?
    where, ask, _, _, check = tasks.training_task(seed, kind)
    here = os.getcwd()
    os.chdir(where)                                      # the harness works in the folder it's started in
    try:
        conversation = [{"role": "user", "content": ask}]
        with contextlib.redirect_stdout(io.StringIO()):
            harness.act(conversation, steps)               # the harness's own loop: our model, its tool, until it hands back
        said = conversation[-1]["content"] if conversation[-1]["role"] == "assistant" else ""
        return harness.assemble_context(conversation, prompt=False), check(said)   # the try as text, as the model saw and wrote it
    finally:
        os.chdir(here)
        shutil.rmtree(where, ignore_errors=True)


def held_out_tasks():                                    # the held-out check: every kind of task, in folders it never trains in
    harness.TEMPERATURE = 0                                # the most likely reply, so the score repeats
    results = [attempt(tasks.TRAINING[n % len(tasks.TRAINING)], seed=10_000 + n) for n in range(40)]
    harness.TEMPERATURE = None                             # tries in training: sampled as its generation settings say
    return results


def score(weights):                                      # any model in the harness, on the 40 held-out tasks: uv run train.py score models/instruct.pt
    global LOG
    name = os.path.basename(weights).removesuffix(".pt")
    LOG = f"runs/score-{name}.txt"
    open(LOG, "w").close()
    harness.load(weights)
    results = held_out_tasks()
    for n, (text, ok) in enumerate(results):
        open(f"runs/score-{name}.said.txt", "a" if n else "w").write(f"{'PASS' if ok else 'FAIL'}  {tasks.TRAINING[n % len(tasks.TRAINING)].__name__}\n{text}\n\n")
        log(f"{'PASS' if ok else 'FAIL'}  {tasks.TRAINING[n % len(tasks.TRAINING)].__name__}")
    log(f"{name} in the harness: {sum(ok for _, ok in results)}/40 held-out tasks")


def rl(rounds=3, group=8):                               # reinforcement learning in the harness: try training tasks, keep what passed
    logging_to("rl")
    harness.load("models/instruct.pt")
    model = harness.model
    done = resume(model, "rl")
    if not done:
        before = sum(ok for _, ok in held_out_tasks())
        log(f"held-out tasks before: {before}/40 passed")
        if not before:                                   # never right: every group of tries would all fail, and there'd be nothing to learn
            sys.exit("RL needs the agent to pass some of the time: improve instruction-tuning first")
        save(model.model, "rl", step=0, passed=[])       # the check before is done: a restart goes straight to training
    harness.TEMPERATURE = None                             # tries sampled as its generation settings say, so they differ
    optimizer, passed = trainable(model.model, lr=1e-6), done.get("passed", [])   # a tenth of instruction-tuning's step: trained on its own words, a big step drifts it
    for step in range(done.get("step", 0) + 1, rounds * len(tasks.TRAINING) + 1):
        kind = tasks.TRAINING[(step - 1) % len(tasks.TRAINING)]   # every kind, once a round, in folders the held-out check never uses
        tries = [attempt(kind, seed=step) for _ in range(group)]
        reward = torch.tensor([float(ok) for _, ok in tries])
        passed.append(reward.mean().item())
        if reward.std() > 0:                             # some passed, some failed: towards the ones that passed
            update(model, optimizer, [text for text, _ in tries], reward.tolist(), most=1536)   # wins weigh 1, losses 0
        log(f"step {step}: {kind.__name__}, {int(reward.sum())}/{group} passed")
        if step % len(tasks.TRAINING) == 0:
            log(f"  round {step // len(tasks.TRAINING)}: {sum(passed[-len(tasks.TRAINING):]) / len(tasks.TRAINING):.0%} of tries passed")
        save(model.model, "rl", step=step, passed=passed)
    log(f"held-out tasks after: {sum(ok for _, ok in held_out_tasks())}/40 passed")
    finish(model, "rl")


if __name__ == "__main__":
    torch.manual_seed(0); random.seed(0)
    os.makedirs("data", exist_ok=True); os.makedirs("models", exist_ok=True); os.makedirs("runs", exist_ok=True)
    {"pre": pretrain, "mid": midtrain, "instruct": instruct, "rl": rl, "score": score}[sys.argv[1]](*sys.argv[2:])
