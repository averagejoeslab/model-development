"""The tasks quark trains on: a fresh folder, a request, a reference solution, and a check of what it left behind.
Instruction-tuning imitates reference solutions that passed their check; RL uses the check as its reward."""
import os, re, sys, json, random, shutil, datetime, tempfile, subprocess

INSTRUCTIONS = "You are quark, an agent. You act through bash, in {where}. Today is {today}."   # quark.py's, word for word
BASH = {"type": "function", "function": {"name": "bash", "description": "Run a shell command",       # quark.py's tool
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}


def system(where=None):                                  # what quark tells the model, word for word, in the folder it works in
    where = where or "/tmp/tmp" + "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789_", k=8))
    return {"role": "system", "content": INSTRUCTIONS.format(where=where, today=datetime.date.today())}


def call(command):
    return {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "bash", "arguments": {"command": command}}}]}


def run(command, where):                                 # really run it, so every result in the data is real
    ran = subprocess.run(command, shell=True, cwd=where, capture_output=True, text=True)
    return (ran.stdout + ran.stderr).strip() or "(no output)"


# Each kind: a request, a reference solution, and a check of the result. Training uses one set of folders (seeds);
# the held-out check uses others, never trained in.

PEOPLE = "ada bo cy dee eli fay gus hal ivy jo kit lu".split()
TOPICS = "server disk cache queue backup login deploy build report invoice".split()
TRAINING = []
kind = TRAINING.append                                   # each kind: (rng, folder) -> (request, commands, answer from their output, check)


def put(where, name, text):
    open(os.path.join(where, name), "w").write(text)


def text(where, name):
    path = os.path.join(where, name)
    return open(path).read() if os.path.isfile(path) else None


def workspace(rng, where):                               # a small, realistic folder: notes, a log, scores, a config, a script
    put(where, "notes.md", "# notes\n" + "".join(f"- {rng.choice(TOPICS)} {rng.choice(['is slow', 'works', 'needs a look', 'was fixed'])}\n"
                                                for _ in range(rng.randint(3, 7))))
    put(where, "todo.md", "".join(f"- {rng.choice(['check', 'update', 'restart', 'clean'])} the {rng.choice(TOPICS)}\n" for _ in range(rng.randint(2, 5))))
    put(where, "server.log", "".join(f"10:{m:02d} {rng.choice(['INFO', 'INFO', 'WARN', 'ERROR'])} {rng.choice(TOPICS)} {rng.choice(['started', 'stopped', 'failed', 'ok'])}\n"
                                     for m in sorted(rng.sample(range(60), rng.randint(5, 12)))))
    put(where, "scores.csv", "name,score\n" + "".join(f"{p},{rng.randint(10, 99)}\n" for p in rng.sample(PEOPLE, rng.randint(3, 6))))
    put(where, "config.json", json.dumps({"name": rng.choice(TOPICS), "port": rng.randint(3000, 9000), "debug": False}, indent=2) + "\n")
    put(where, "deploy.sh", "#!/bin/sh\necho deploying\n")


def lines_of(where, name):
    return text(where, name).splitlines()


@kind
def last_lines(rng, where):
    n, out = rng.randint(2, 4), rng.choice(["recent.txt", "tail.txt", "latest.log"])
    want = lines_of(where, "server.log")[-n:]
    return (rng.choice([f"Save the last {n} lines of server.log to {out}.", f"Put the {n} most recent lines of server.log in {out}."]),
            [f"tail -n {n} server.log > {out}", f"cat {out}"], lambda o: f"Done: {out} has the last {n} lines:\n{o[-1]}",
            lambda said: (text(where, out) or "").splitlines() == want)


@kind
def log_levels(rng, where):
    out = rng.choice(["levels.txt", "kinds.txt"])
    want = sorted({line.split()[1] for line in lines_of(where, "server.log")})
    return (f"List each level that appears in server.log (INFO, WARN, ...), once each, in {out}.",
            [f"awk '{{print $2}}' server.log | sort -u > {out}", f"cat {out}"], lambda o: f"Done: {out} lists " + ", ".join(o[-1].split()) + ".",
            lambda said: (text(where, out) or "").split() == want)


@kind
def count_errors(rng, where):
    out = rng.choice(["errors.txt", "error_count.txt"])
    want = sum("ERROR" in line for line in lines_of(where, "server.log"))
    return (rng.choice([f"How many ERROR lines are in server.log? Save the number to {out}.", f"Count the errors in server.log and put the number in {out}."]),
            [f"grep -c ERROR server.log > {out}", f"cat {out}"], lambda o: f"Done: {out} says {o[-1]}.",
            lambda said: (text(where, out) or "").strip() == str(want))


@kind
def csv_names(rng, where):
    out = rng.choice(["names.txt", "people.txt"])
    want = [line.split(",")[0] for line in lines_of(where, "scores.csv")[1:]]
    return (f"Put the names from scores.csv in {out}, one per line, without the header.",
            [f"tail -n +2 scores.csv | cut -d, -f1 > {out}", f"cat {out}"], lambda o: f"Done: {out} has " + ", ".join(o[-1].split()) + ".",
            lambda said: (text(where, out) or "").split() == want)


@kind
def csv_total(rng, where):
    out = rng.choice(["total.txt", "sum.txt"])
    want = sum(int(line.split(",")[1]) for line in lines_of(where, "scores.csv")[1:])
    return (rng.choice([f"Add up the scores in scores.csv and save the total to {out}.", f"What do the scores in scores.csv add up to? Write it to {out}."]),
            [f"awk -F, 'NR > 1 {{s += $2}} END {{print s}}' scores.csv > {out}", f"cat {out}"], lambda o: f"Done: the total is {o[-1]}, in {out}.",
            lambda said: (text(where, out) or "").strip() == str(want))


@kind
def uppercase_copy(rng, where):
    source, out = rng.choice(["notes.md", "todo.md"]), rng.choice(["LOUD.md", "upper.md"])
    want = text(where, source).upper()
    return (f"Make an uppercase copy of {source} called {out}.", [f"tr a-z A-Z < {source} > {out}", f"head -n 3 {out}"],
            lambda o: f"Done: {out} is {source} in capitals; it starts:\n{o[-1]}", lambda said: text(where, out) == want)


@kind
def drop_lines(rng, where):
    topic = rng.choice([t for t in TOPICS if t in text(where, "notes.md")] or TOPICS)
    want = [line for line in lines_of(where, "notes.md") if topic not in line]
    return (rng.choice([f"Remove every line that mentions {topic} from notes.md.", f"Delete the {topic} lines from notes.md."]),
            [f"sed -i '/{topic}/d' notes.md", "cat notes.md"], lambda o: f"Done: notes.md no longer mentions {topic}:\n{o[-1]}",
            lambda said: lines_of(where, "notes.md") == want)


@kind
def nested_folders(rng, where):
    path = "/".join(rng.sample(["src", "docs", "data", "old", "2026", "tmp", "lib"], 3))
    return (f"Create the folders {path}.", [f"mkdir -p {path}", f"ls -d {path}"], lambda o: f"Done: {o[-1]} exists now.",
            lambda said: os.path.isdir(os.path.join(where, path)))


@kind
def config_port(rng, where):
    out = rng.choice(["port.txt", "port"])
    want = json.load(open(os.path.join(where, "config.json")))["port"]
    return (f"Save the port from config.json to {out}.",
            [f"""python3 -c "import json; print(json.load(open('config.json'))['port'])" > {out}""", f"cat {out}"],
            lambda o: f"Done: the port is {o[-1]}, saved in {out}.", lambda said: (text(where, out) or "").strip() == str(want))


@kind
def config_debug(rng, where):
    def check(said):
        try:
            config = json.load(open(os.path.join(where, "config.json")))
        except ValueError:
            return False
        return config.get("debug") is True and {k: v for k, v in config.items() if k != "debug"} == before
    before = {k: v for k, v in json.load(open(os.path.join(where, "config.json"))).items() if k != "debug"}
    return (rng.choice(["Turn debug on in config.json.", "Set debug to true in config.json, and leave the rest alone."]),
            ["""python3 -c "import json; c = json.load(open('config.json')); c['debug'] = True; json.dump(c, open('config.json', 'w'), indent=2)" """.strip(), "cat config.json"],
            lambda o: f"Done: config.json now has debug set to true:\n{o[-1]}", check)


@kind
def markdown_lines(rng, where):
    out = rng.choice(["md_lines.txt", "lines.txt"])
    want = sum(len(lines_of(where, f)) for f in os.listdir(where) if f.endswith(".md"))
    return (f"How many lines do the .md files have, all together? Save the number to {out}.",
            [f"cat *.md | wc -l > {out}", f"cat {out}"], lambda o: f"Done: {o[-1]} lines in all, saved in {out}.",
            lambda said: (text(where, out) or "").strip() == str(want))


@kind
def backup_csv(rng, where):
    folder_name = rng.choice(["backup", "saved", "copies"])
    want = sorted(f for f in os.listdir(where) if f.endswith(".csv"))
    return (f"Back up every .csv file into a folder called {folder_name}, keeping the originals.",
            [f"mkdir -p {folder_name}", f"cp *.csv {folder_name}/", f"ls {folder_name}"],
            lambda o: f"Done: {folder_name} has " + ", ".join(o[-1].split()) + "; the originals are still here.",
            lambda said: os.path.isdir(os.path.join(where, folder_name)) and sorted(os.listdir(os.path.join(where, folder_name))) == want
            and all(os.path.isfile(os.path.join(where, f)) for f in want))


@kind
def nth_line(rng, where):
    n, out = rng.randint(2, 4), rng.choice(["line.txt", "picked.txt"])
    want = lines_of(where, "server.log")[n - 1]
    return (f"Save line {n} of server.log to {out}.", [f"sed -n '{n}p' server.log > {out}", f"cat {out}"],
            lambda o: f"Done: {out} says: {o[-1]}", lambda said: (text(where, out) or "").strip() == want)


@kind
def add_title(rng, where):
    title = rng.choice(["Plan", "Today", "Team notes", "Ops"])
    want = [f"# {title}"] + lines_of(where, "todo.md")
    return (f"Add a title line '# {title}' at the top of todo.md.", [f"sed -i '1i # {title}' todo.md", "head -n 2 todo.md"],
            lambda o: f"Done: todo.md now starts with # {title}.", lambda said: lines_of(where, "todo.md") == want)


@kind
def make_executable(rng, where):
    return (rng.choice(["Make deploy.sh executable.", "deploy.sh won't run: make it executable."]), ["chmod +x deploy.sh", "ls -l deploy.sh"],
            lambda o: "Done: deploy.sh is executable now.", lambda said: os.access(os.path.join(where, "deploy.sh"), os.X_OK))


def unchanged(where):                                    # questions: answer, and change nothing
    before = {f: text(where, f) for f in os.listdir(where)}
    return lambda: {f: text(where, f) for f in os.listdir(where)} == before


@kind
def ask_port(rng, where):
    port, same = json.load(open(os.path.join(where, "config.json")))["port"], unchanged(where)
    return ("What port does config.json set? Just tell me, don't touch it.", ["cat config.json"], lambda o: f"It sets port {port}.",
            lambda said: str(port) in said and same())


@kind
def ask_errors(rng, where):
    n, same = sum("ERROR" in line for line in lines_of(where, "server.log")), unchanged(where)
    return ("How many ERROR lines does server.log have? Just tell me.", ["grep -c ERROR server.log"], lambda o: f"server.log has {o[-1]} ERROR lines.",
            lambda said: re.search(rf"\b{n}\b", said) is not None and same())


@kind
def ask_top(rng, where):
    rows = [line.split(",") for line in lines_of(where, "scores.csv")[1:]]
    best, same = [r[0] for r in rows if int(r[1]) == max(int(r[1]) for r in rows)], unchanged(where)   # a tie: any of them
    return ("Who has the highest score in scores.csv? Don't change anything.", ["tail -n +2 scores.csv | sort -t, -k2 -nr | head -n 1"],
            lambda o: f"{o[-1].split(',')[0]}, with {o[-1].split(',')[1]}.", lambda said: any(b in said for b in best) and same())


@kind
def ask_last(rng, where):
    last, same = lines_of(where, "todo.md")[-1], unchanged(where)
    return ("What's the last item in todo.md? Don't change anything.", ["tail -n 1 todo.md"], lambda o: f"The last item is: {o[-1].lstrip('- ')}",
            lambda said: last.lstrip("- ") in said and same())


BUGS = [   # (function and its argument, the bug, the fix, the sed that makes it, a test: call and answer, what was wrong)
    ("double(x)", "return x * 3", "return x * 2", "s/x \\* 3/x * 2/", "double(4)", "8", "it multiplied by 3 instead of 2"),
    ("last(items)", "return items[0]", "return items[-1]", "s/items\\[0\\]/items[-1]/", "last([1, 2, 3])", "3", "it returned the first item, not the last"),
    ("is_even(n)", "return n % 2 == 1", "return n % 2 == 0", "s/== 1/== 0/", "is_even(4)", "True", "it checked for a remainder of 1, which means odd"),
    ("smallest(numbers)", "return max(numbers)", "return min(numbers)", "s/max/min/", "smallest([3, 1, 2])", "1", "it used max instead of min"),
    ("square(x)", "return x + x", "return x * x", "s/x + x/x * x/", "square(3)", "9", "it added x to itself instead of multiplying"),
    ("count_words(text)", "return len(text)", "return len(text.split())", "s/len(text)/len(text.split())/", "count_words('a b c')", "3", "it counted characters, not words"),
    ("greet(name)", 'return "Hi " + name', 'return "Hello " + name', "s/Hi /Hello /", "greet('Ann')", "'Hello Ann'", "it said Hi instead of Hello"),
    ("gap(a, b)", "return a + b", "return a - b", "s/a + b/a - b/", "gap(7, 2)", "5", "it added instead of subtracting"),
    ("rest(items)", "return items[:-1]", "return items[1:]", "s/items\\[:-1\\]/items[1:]/", "rest([1, 2, 3])", "[2, 3]", "it dropped the last item instead of the first"),
    ("tidy(text)", "return text.split()", "return text.strip()", "s/split()/strip()/", "tidy('  hi ')", "'hi'", "it split the text into words instead of trimming it"),
    ("biggest(numbers)", "return sorted(numbers)[0]", "return sorted(numbers)[-1]", "s/\\[0\\]/[-1]/", "biggest([3, 9, 4])", "9", "it took the smallest after sorting, not the largest"),
    ("half(x)", "return x // 2", "return x / 2", "s#x // 2#x / 2#", "half(5)", "2.5", "it used whole-number division"),
    ("initials(name)", "return name[0]", "return name[0].upper()", "s/name\\[0\\]/name[0].upper()/", "initials('ada')", "'A'", "it didn't capitalise"),
]
STRAIGHT = [   # (function and its argument, its right code, a test: call and answer): nothing wrong with these
    ("triple(x)", "return x * 3", "triple(2)", "6"), ("first(items)", "return items[0]", "first([5, 6])", "5"),
    ("is_odd(n)", "return n % 2 == 1", "is_odd(3)", "True"), ("shout(text)", "return text.upper()", "shout('a')", "'A'"),
]


@kind
def fix_code(rng, where):                                # failing tests: look, run them, read the code, fix it, run them again
    function, bug, fix, sed, test, answer, why = rng.choice(BUGS)
    module = rng.choice(["helpers", "mathlib", "util", "core", "tools"])
    put(where, f"{module}.py", f"def {function}:\n    {bug}\n")
    tests = (f"import unittest\nfrom {module} import {function.split('(')[0]}\n\n\nclass Test(unittest.TestCase):\n"
             f"    def test_it(self):\n        self.assertEqual({test}, {answer})\n\n\nif __name__ == \"__main__\":\n    unittest.main()\n")
    put(where, f"test_{module}.py", tests)

    def check(said):
        shutil.rmtree(os.path.join(where, "__pycache__"), ignore_errors=True)   # judge the code as it is, not a stale cache
        ok = subprocess.run([sys.executable, "-m", "unittest", "-q"], cwd=where, capture_output=True).returncode == 0
        return ok and text(where, f"test_{module}.py") == tests
    return (rng.choice(["The tests are failing. Can you fix them?", "Make the tests pass.", "Something's broken: the tests fail. Fix the code."]),
            ["ls", "python3 -B -m unittest -q", f"cat {module}.py", f"sed -i '{sed}' {module}.py", "python3 -B -m unittest -q"],   # -B: no stale cache
            lambda o: f"Fixed: in {module}.py, {why}. It now says `{fix}`, and the tests pass.", check)


@kind
def fix_one_of_two(rng, where):                          # two functions, two tests, one bug: find which before fixing
    function, bug, fix, sed, test, answer, why = rng.choice(BUGS)
    other, right, other_test, other_answer = rng.choice(STRAIGHT)
    module = rng.choice(["helpers", "mathlib", "util", "core", "tools", "funcs"])
    names = [function.split("(")[0], other.split("(")[0]]
    put(where, f"{module}.py", f"def {function}:\n    {bug}\n\n\ndef {other}:\n    {right}\n")
    tests = (f"import unittest\nfrom {module} import {', '.join(names)}\n\n\nclass Test(unittest.TestCase):\n"
             f"    def test_{names[0]}(self):\n        self.assertEqual({test}, {answer})\n\n"
             f"    def test_{names[1]}(self):\n        self.assertEqual({other_test}, {other_answer})\n\n\nif __name__ == \"__main__\":\n    unittest.main()\n")
    put(where, f"test_{module}.py", tests)

    def check(said):
        shutil.rmtree(os.path.join(where, "__pycache__"), ignore_errors=True)
        ok = subprocess.run([sys.executable, "-m", "unittest", "-q"], cwd=where, capture_output=True).returncode == 0
        return ok and text(where, f"test_{module}.py") == tests
    return (rng.choice([f"One of the tests in test_{module}.py fails. Find the bug and fix it.", f"test_{module}.py has a failure. Fix the code it tests."]),
            ["ls", "python3 -B -m unittest -q", f"cat {module}.py", f"sed -i '{sed}' {module}.py", "python3 -B -m unittest -q"],
            lambda o: f"Fixed: {names[1]} was fine; in {names[0]}, {why}. It now says `{fix}`, and both tests pass.", check)


@kind
def archive_logs(rng, where):                            # two actions: a new folder, then the files into it
    folder_name = rng.choice(["archive", "old_logs", "history"])
    want = sorted(f for f in os.listdir(where) if f.endswith(".log"))
    return (rng.choice([f"Put the log files into a new folder named {folder_name}.", f"Create {folder_name} and move the .log files there."]),
            [f"mkdir -p {folder_name}", f"mv *.log {folder_name}/", f"ls {folder_name}"],
            lambda o: f"Done: {folder_name} now holds " + ", ".join(o[-1].split()) + ", and none are left here.",
            lambda said: os.path.isdir(os.path.join(where, folder_name)) and sorted(os.listdir(os.path.join(where, folder_name))) == want
            and not any(f.endswith(".log") for f in os.listdir(where)))


@kind
def which_mentions(rng, where):                          # find it, then write what you found
    topic = rng.choice([t for t in TOPICS if any(t in (text(where, f) or "") for f in ("notes.md", "todo.md"))] or TOPICS)
    out = rng.choice(["where.txt", "found.txt"])
    have = {f for f in ("notes.md", "todo.md") if topic in (text(where, f) or "")}
    return (f"Which .md file talks about the {topic}? Save that filename in {out}.",
            [f"grep -l {topic} *.md", f"grep -l {topic} *.md | head -n 1 > {out}", f"cat {out}"],
            lambda o: f"Done: {o[-1]} mentions the {topic}; its name is in {out}.", lambda said: (text(where, out) or "").strip() in have)


@kind
def most_lines(rng, where):                              # measure every file, then pick
    out = rng.choice(["longest.txt", "most.txt"])
    counts = {f: len(lines_of(where, f)) for f in os.listdir(where) if f.endswith(".md")}
    best = {f for f, n in counts.items() if n == max(counts.values())}
    return (f"Of the .md files, which has the most lines? Write its name in {out}.",
            ["wc -l *.md", f"wc -l *.md | grep -v total | sort -n | tail -n 1 | awk '{{print $2}}' > {out}", f"cat {out}"],
            lambda o: f"Done: {o[-1]} has the most lines; its name is in {out}.", lambda said: (text(where, out) or "").strip() in best)


@kind
def append_todo(rng, where):
    item = f"- {rng.choice(['check', 'update', 'restart', 'clean'])} the {rng.choice(TOPICS)}"
    want = lines_of(where, "todo.md") + [item]
    return (rng.choice([f"Put '{item}' at the bottom of todo.md.", f"Append the item '{item}' to todo.md."]),
            [f"echo '{item}' >> todo.md", "tail -n 2 todo.md"], lambda o: f"Done: todo.md ends with {item}.",
            lambda said: lines_of(where, "todo.md") == want)


@kind
def copy_then_edit(rng, where):                          # copy, then change the copy and only the copy
    name, port = rng.choice(["staging.json", "local.json", "test.json"]), rng.randint(3000, 9000)
    before = json.load(open(os.path.join(where, "config.json")))

    def check(said):
        try:
            copy = json.load(open(os.path.join(where, name)))
        except (ValueError, OSError):
            return False
        return json.load(open(os.path.join(where, "config.json"))) == before and copy == {**before, "port": port}
    return (f"Copy config.json to {name} and set the port in {name} to {port}; leave config.json alone.",
            [f"cp config.json {name}", f"""python3 -c "import json; c = json.load(open('{name}')); c['port'] = {port}; json.dump(c, open('{name}', 'w'), indent=2)" """.strip(), f"cat {name}"],
            lambda o: f"Done: {name} is a copy of config.json with port {port}; config.json is unchanged.", check)


READS = {"cat", "head", "tail", "ls", "grep", "wc", "python3"}   # the commands that look


def training_task(seed, which=None):                     # a fresh folder, and one task in it
    rng, where = random.Random(seed), tempfile.mkdtemp()
    workspace(rng, where)
    ask, commands, answer, check = (which or rng.choice(TRAINING))(rng, where)
    return where, ask, commands, answer, check


def tool_session():                                      # a verified demonstration: a request, the commands, their real results, an answer
    while True:
        where, ask, commands, answer, check = training_task(random.random())
        messages, outs = [system(where), {"role": "user", "content": ask}], []
        if random.random() < 0.5 and commands[0] not in ("ls", "cat config.json"):
            commands = ["ls"] + commands                 # sometimes, look around first
        for command in commands:
            outs.append(run(command, where))
            messages += [call(command), {"role": "tool", "content": outs[-1]}]
        reply = answer(outs)
        passed = check(reply) and commands[-1].split()[0] in READS   # kept only if it passes, and looked before it answered
        shutil.rmtree(where)
        if passed:
            return messages + [{"role": "assistant", "content": reply}]
