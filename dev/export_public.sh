#!/usr/bin/env bash
# Writes public/: exactly the files a push to GitHub publishes (the committed tree of the current
# branch, submodules included), to look through before pushing. Everything else in the project
# folder (the game dump, data/, logs, captures, tools/, dist/) is the local workshop and stays
# out: .gitignore keeps it out of every commit. public/ is ignored too and rebuilt each time.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
dest=public
mkdir -p "$dest"
find "$dest" -mindepth 1 -delete
git archive --format=tar HEAD | tar -x -C "$dest"
git submodule foreach --quiet --recursive \
    'git archive --format=tar --prefix="$displaypath/" HEAD | tar -x -C "$toplevel/'"$dest"'"'
# Nothing of the player's may be published.
if find "$dest" \( -iname 'eboot.bin' -o -iname '*.sfo' -o -iname '*.pkg' -o -iname '*.prx' -o -iname '*.sprx' \) | grep -q .; then
    echo 'A game file is committed: remove it from git before pushing' >&2
    exit 1
fi
echo "public/: $(find "$dest" -type f | wc -l) files, $(du -sh "$dest" | cut -f1) (commit $(git rev-parse --short HEAD))"
