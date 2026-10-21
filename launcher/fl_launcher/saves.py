# SPDX-License-Identifier: GPL-2.0-or-later
"""The player's saves as the launcher shows them: each save's sce_sys/param.bin (the console's
SceSaveDataParam: title 128 bytes, subtitle 128, detail 1024, user param, then mtime at 1288) holds
what the game wrote for it. First Light writes the mission and the story progress in the
subtitle, in the game's language: "Ненависть в Сиэтле - 27%", "(Автосохранение) ..." for an
autosave. The save data itself (dis.dat) is left alone."""
import datetime
import re
import struct
from pathlib import Path

PARAM_SIZE = 1328
PREFIX = re.compile(r'^\([^)]*\)\s*')                       # "(Автосохранение) ", "(Autosave) "
PROGRESS = re.compile(r'^(?P<mission>.*?)\s*[-–]\s*(?P<percent>\d{1,3})\s*%\s*$')


def _text(raw):
    return raw.split(b'\0', 1)[0].decode('utf-8', 'replace').strip()


def read_param(path):
    """{'subtitle', 'detail', 'mtime'} of one save's param.bin; None when it is not one."""
    data = Path(path).read_bytes()
    if len(data) < PARAM_SIZE:
        return None
    mtime = struct.unpack_from('<q', data, 1288)[0]
    return {'subtitle': _text(data[128:256]), 'detail': _text(data[256:1280]),
            'mtime': mtime if 0 < mtime < 2 ** 33 else Path(path).stat().st_mtime}


def parse(subtitle):
    """(mission, percent or None) from a save's subtitle."""
    line = PREFIX.sub('', subtitle)
    match = PROGRESS.match(line)
    if not match:
        return line, None
    return match['mission'], min(100, int(match['percent']))


def list_saves(user_dir, title_id):
    """The saves of `title_id` in <user>/savedata/<user id>/<title id>/, newest first:
    {'slot', 'auto', 'mission', 'percent', 'mtime'}."""
    saves = []
    root = Path(user_dir) / 'savedata'
    if not title_id or not root.is_dir():
        return saves
    for param in root.glob(f'*/{title_id}/*.sce_sys/param.bin'):
        try:
            values = read_param(param)
        except OSError:
            continue
        if not values or not values['subtitle']:
            continue
        slot = param.parent.name[:-len('.sce_sys')]
        mission, percent = parse(values['subtitle'])
        saves.append({'slot': slot, 'auto': 'auto' in slot.lower(), 'mission': mission,
                      'percent': percent, 'mtime': values['mtime']})
    saves.sort(key=lambda save: save['mtime'], reverse=True)
    return saves


def when(mtime, texts, now=None):
    """'сегодня в 22:54', 'вчера в 19:02', '21 октября, 19:02' (texts: i18n.Texts)."""
    moment = datetime.datetime.fromtimestamp(mtime)
    today = (now or datetime.datetime.now()).date()
    clock = moment.strftime('%H:%M')
    if moment.date() == today:
        return texts('save_today', time=clock)
    if moment.date() == today - datetime.timedelta(days=1):
        return texts('save_yesterday', time=clock)
    month = texts('months').split(',')[moment.month - 1]
    return texts('save_date', day=moment.day, month=month, time=clock)
