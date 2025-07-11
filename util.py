import textwrap
import re

def prepare_prompt(prompt: str) -> str:

    cws = re.compile(r"\s+")
    tmp = textwrap.dedent(prompt).replace("\n", " ")
    return cws.sub(" ", tmp).strip()