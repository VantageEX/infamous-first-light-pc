# SPDX-License-Identifier: GPL-2.0-or-later
"""Core switches a game runs with unless the environment already sets them (run.py applies them
once it knows the dump's title id): whoever starts the core, the launcher or run.py by hand, gets
the same defaults, and only the game they were measured on gets them."""

# inFAMOUS First Light (CUSA00897 EU, CUSA00575 US).
FIRST_LIGHT = {
    # Precise GPU readbacks. The core defaults to relaxed, where the CPU never sees what the GPU
    # wrote: First Light's emissive exposure sticks at its -9.5 EV floor, so the neon dash trail
    # is ~1000 times too dim and the city stays dark after loading a save.
    'TLG_READBACKS': '2',
    'TLG_EAGER_READBACK': '0',     # on, smoke and glow flicker in precise mode (iss_pc)
    'TLG_WRITE_MERGE': '1',        # game memcpy/memset into GPU memory without a GPU wait
    'TLG_PIPELINE_BINARIES': '0',  # VK_KHR_pipeline_binary crashes on the NVIDIA 595 driver
    'TLG_GC_BUDGET_MB': '3500',    # texture cache budget (iss_pc: 5600 runs out of 6 GB of VRAM)
    'TLG_FRAME_STATS': '1',        # FPS in the log every 5 s
    # The body of First Light's particle lists (both buffer sets), read by the GPU only: their
    # read faults stay out of precise mode (a little FPS). Their first page (the list header)
    # and the emitter counters after them stay precise: without those part of the dash effects
    # vanish. Fixed addresses: the game allocates them the same way every start (measured on
    # CUSA00897).
    'TLG_RELAXED_RANGES': '106e159000-106e258000,106ff59000-1070058000',
    # The light glare reads two occlusion queries a frame. Mode 3 gives it 0/0 (yellow screens
    # and buildings), 2 a negative count (white-outs), 4 (iss_pc's real queries, the core's
    # default) yellow flickers and host waits. 5 is this port's fixed step.
    'TLG_OCCLUSION_MODE': '5',
}

GAMES = {'CUSA00897': FIRST_LIGHT, 'CUSA00575': FIRST_LIGHT}


def apply(env, title_id):
    """Sets the defaults of the game `title_id` that `env` does not have; returns their names."""
    applied = []
    for key, value in GAMES.get(title_id or '', {}).items():
        if key not in env:
            env[key] = value
            applied.append(key)
    return applied
