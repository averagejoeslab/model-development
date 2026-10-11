# /// script
# dependencies = ["torch", "numpy", "regex", "safetensors", "huggingface_hub", "jinja2"]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""quark, the smallest agent: building-agents' quark.py, with our model in it. uv run quark.py models/rl.pt"""
import subprocess, os, re, sys, json, datetime
from model import Release

model = None                                             # set by load(): Qwen's chat format, with the numbers we trained
TEMPERATURE = 0                                          # the most likely reply; training samples instead
bash = {"type": "function", "function": {"name": "bash", "description": "Run a shell command",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}


def load(weights):
    global model
    model = Release("Qwen/Qwen3-0.6B", weights)


def capture_input():
    return input("\n> ")


def assemble_context(conversation, prompt=True):         # the conversation, laid out in the chat template our model learned
    instructions = f"You are quark, an agent. You act through bash, in {os.getcwd()}. Today is {datetime.date.today()}."
    return model.chat([{"role": "system", "content": instructions}, *conversation], tools=[bash], prompt=prompt)


def request_response(context):                           # our model, not an API: it writes the reply one token at a time
    return model.generate(context, most=512, temperature=TEMPERATURE)


def handle_output(response):                             # its words are shown; each <tool_call> it wrote is run
    print("< " + re.sub(r"<tool_call>.*?(</tool_call>|$)", "", response, flags=re.S).strip().replace("\n", "\n  "))
    tool_results = []
    for call in re.findall(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", response, re.S):
        try:
            command = json.loads(call)["arguments"]["command"]
        except (ValueError, KeyError, TypeError):
            tool_results.append({"role": "tool", "content": "[not a tool call quark can read]"})
            continue
        print(f"$ {command}")
        try:                                             # nobody's watching in training, so a command gets a minute
            ran = subprocess.run(command, shell=True, capture_output=True, text=True, errors="replace", stdin=subprocess.DEVNULL, timeout=60)
            result = ran.stdout + ran.stderr or "(no output)"
        except subprocess.TimeoutExpired:
            result = "[stopped: it ran for over a minute]"
        tool_results.append({"role": "tool", "content": result[:4000]})
    return tool_results


def act(conversation, steps=None):                       # the inner loop: go again while a tool ran; in training, at most `steps`
    while steps is None or steps > 0:
        response = request_response(assemble_context(conversation))
        conversation.append({"role": "assistant", "content": response})
        tool_results = handle_output(response)
        if not tool_results:
            return                                       # done: hand back to the person
        conversation += tool_results                     # a tool ran: go again
        steps = steps and steps - 1


def control_flow():
    conversation = []
    while True:
        conversation.append({"role": "user", "content": capture_input()})
        act(conversation)


if __name__ == "__main__":
    load(sys.argv[1] if len(sys.argv) > 1 else "models/rl.pt")
    control_flow()
