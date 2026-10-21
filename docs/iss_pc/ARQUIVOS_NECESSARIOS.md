# Arquivos do jogo necessários / Required game files

[Português](#arquivos-do-jogo-necessários) · [English](#required-game-files)

## Arquivos do jogo necessários

Tudo vem do **seu** PS4 (GoldHEN + Itemzflow). Nada disso vai no download. Use **inFAMOUS Second Son CUSA00223 (EUA), jogo base 1.00 com o update 1.07**, montado numa pasta só, com o jogo base e o update por cima. São ~33 GB (27 GB do jogo base + 6,4 GB do update).

| Arquivo | Vem de | Para quê |
|---|---|---|
| `eboot.bin` | update 1.07 | o código do jogo (tem que ser o do 1.07) |
| `sce_module/libc.prx`, `sce_module/libSceFios2.prx` | update 1.07 | bibliotecas do próprio jogo (a `libc.prx` do 1.00 não serve) |
| `sce_sys/` (`param.sfo`, `playgo-chunk.dat`, `icon0.png`, `pic1.png`) | jogo base (+ `param.sfo` do update) | metadados, instalação e a arte do launcher |
| `art/` (`cache/`, `movies/`, `ui/`) | base + update | conteúdo do jogo (o update substitui parte de `art/cache` e `art/movies`) |
| `playgo.pgm`, `sce_discmap.plt`, `sce_discmap_patch.plt` | base + update | mapa de arquivos e instalação |

**Versões do jogo:** testado com **CUSA00223 (1.00 + update 1.07)**; o launcher só aceita a pasta com o update 1.07. Há relato de que o CUSA00004 também inicia (ainda não testado a fundo). Idiomas: o launcher oferece só os idiomas que o `sce_sys/playgo-chunk.dat` do seu dump marca, entre os 30 idiomas do PS4 (um dump europeu traz, por exemplo, italiano, polonês, alemão e espanhol da Espanha); sem esse arquivo (o FTP não mostra) ele oferece os 6 do CUSA00223 (inglês, português do Brasil, espanhol latino, francês do Canadá, turco e russo), e cada um só funciona se os arquivos do seu dump tiverem o idioma.

**Requisitos do PC:** Windows 10 (1903+)/11 64 bits ou Linux x86-64 (glibc 2.39+), placa de vídeo com Vulkan 1.3 (6 GB de VRAM recomendados) e ~33 GB livres para o jogo. O `sce_sys/pic1.png` do seu dump aparece como fundo do launcher (lido da sua pasta, nunca distribuído).

**Do firmware do console:** nada. Este jogo não usa nenhum módulo do sistema do PS4 (`libSceNgs2`, `libSceSecure` e `libSceJobManager` não são importados). A pasta `system_lib/` do pacote fica vazia.

**Como montar a pasta:**
1. No Itemzflow, faça o dump do jogo (`CUSA00223`) para um pendrive exFAT. Para o update (`CUSA00223-patch`), dá para usar o dump do Itemzflow ou o FTP do GoldHEN com o jogo aberto (`/mnt/sandbox/pfsmnt/CUSA00223-patch0`).
2. No PC, copie a pasta do jogo base e depois copie o conteúdo do update **por cima**, substituindo os arquivos. Também dá para usar `scripts/make_game_view.py <base> <update> <saída> [param.sfo do 1.07]`, que monta a pasta com links, sem ocupar espaço.
3. O `sce_sys` do update não traz o `param.sfo` da versão 1.07. No PS4 ele fica em `/system_data/priv/appmeta/CUSA00223/param.sfo` (pegue pelo FTP do GoldHEN e copie para o `sce_sys/` da pasta montada). Com o `make_game_view.py`, passe esse arquivo como 4º argumento ou ponha numa pasta `patch-meta/` dentro do jogo base (ex.: `dump/patch-meta/param.sfo`): a visão usa o `param.sfo` de maior versão. Sem ele o jogo abre, mas o launcher avisa e os mods feitos para o 1.07 não aparecem.
4. Confira os tamanhos. Nenhum arquivo deste jogo passa de 4 GB, então o FTP do GoldHEN também serve.

## Required game files

Everything comes from **your own** PS4 (GoldHEN + Itemzflow). None of it is in the download. Use **inFAMOUS Second Son CUSA00223 (US), base game 1.00 with update 1.07**, merged into one folder, with the base game and the update on top. About 33 GB (27 GB base game + 6.4 GB update).

| File | From | Purpose |
|---|---|---|
| `eboot.bin` | update 1.07 | the game code (must be the 1.07 one) |
| `sce_module/libc.prx`, `sce_module/libSceFios2.prx` | update 1.07 | the game's own libraries (the 1.00 `libc.prx` does not work) |
| `sce_sys/` (`param.sfo`, `playgo-chunk.dat`, `icon0.png`, `pic1.png`) | base game (+ the update's `param.sfo`) | metadata, install info and the launcher art |
| `art/` (`cache/`, `movies/`, `ui/`) | base + update | game content (the update replaces part of `art/cache` and `art/movies`) |
| `playgo.pgm`, `sce_discmap.plt`, `sce_discmap_patch.plt` | base + update | file map and install info |

**Game versions:** tested with **CUSA00223 (1.00 + update 1.07)**; the launcher only accepts the folder with the 1.07 update. There is a report that CUSA00004 also boots (not tested in depth yet). Languages: the launcher only offers the languages your dump's `sce_sys/playgo-chunk.dat` lists, out of the PS4's 30 (a European dump has, for example, Italian, Polish, German and Spain Spanish); without that file (FTP does not show it) it offers the 6 of CUSA00223 (English, Brazilian Portuguese, Latin American Spanish, Canadian French, Turkish and Russian), and each only works if your dump's files have that language.

**PC requirements:** 64-bit Windows 10 (1903+)/11 or x86-64 Linux (glibc 2.39+), a graphics card with Vulkan 1.3 (6 GB of VRAM recommended) and ~33 GB free for the game. The `sce_sys/pic1.png` from your dump is used as the launcher's background (read from your folder, never distributed).

**From the console firmware:** nothing. This game uses no PS4 system module (`libSceNgs2`, `libSceSecure` and `libSceJobManager` are not imported). The package's `system_lib/` folder stays empty.

**How to build the folder:**
1. In Itemzflow, dump the game (`CUSA00223`) to an exFAT USB drive. For the update (`CUSA00223-patch`), use either an Itemzflow dump or GoldHEN's FTP with the game running (`/mnt/sandbox/pfsmnt/CUSA00223-patch0`).
2. On the PC, copy the base game folder, then copy the update's contents **over it**, replacing files. Or use `scripts/make_game_view.py <base> <update> <out> [1.07 param.sfo]`, which builds the folder with links and uses no extra space.
3. The update's `sce_sys` does not carry the 1.07 `param.sfo`. On the PS4 it is at `/system_data/priv/appmeta/CUSA00223/param.sfo` (get it through GoldHEN's FTP and copy it into the merged folder's `sce_sys/`). With `make_game_view.py`, pass that file as the 4th argument or put it in a `patch-meta/` folder inside the base game (e.g. `dump/patch-meta/param.sfo`): the view uses the newest `param.sfo`. Without it the game starts, but the launcher warns and mods made for 1.07 are not listed.
4. Check the sizes. No file of this game is over 4 GB, so GoldHEN's FTP works too.
