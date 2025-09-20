#!/usr/bin/env python3
import shlex, sys

def strip_zsh_line(line: str) -> str:
    if line.startswith('#!'):
        return line.rstrip('\n')
    if line.lstrip().startswith('#'):
        return ''
    tokens = shlex.split(line, comments=True, posix=True)
    return ' '.join(tokens) if tokens else ''

src = sys.argv[1]
dst = sys.argv[2]
with open(src) as f, open(dst, 'w') as g:
    for ln in f:
        out = strip_zsh_line(ln)
        if out:
            g.write(out + '\n')
