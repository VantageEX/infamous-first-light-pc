# inFAMOUS Second Son PC: roadmap

Port de inFAMOUS Second Son (Sucker Punch, PS4, CUSA00223, base 1.00 + update 1.07, SDK 4.00).
Atualizado em 15/10/2026. O estado de quem joga está no [README](../README.md#status); cada item
aberto vira uma issue com o tempo.

## Onde estamos

**Jogável, em desenvolvimento:** boot, título, intro e gameplay (testado até "Go to Longhouse"),
com som próprio (AudioOut + Ajm, sem arquivo do firmware), saves, HUD correto, sensor de movimento
(real ou simulado), touchpad simulado e partículas da GPU com a leitura precisa (padrão).
20–30 FPS típicos numa RTX 2060, instável. Launcher Qt como padrão (com o Tk de reserva), saves e dados
configuráveis, 12 correções experimentais (BP01–BP12) e o contador de oclusão corrigido (faróis e brilho
de luzes), confirmado no Windows e no Linux.

## Feito

| Item | Detalhe |
|---|---|
| Boot | `mov reg, fs:[0]` em qualquer registrador (TLS); os 15 imports só deste jogo (`runtime_infamous.c`) |
| Sensor de movimento | giroscópio real pela SDL3 e "modo sensor" simulado (Share ou L3+R3; teclado T/G/Y) |
| Analógicos | zona morta de 10% (`ISS_STICK_DEADZONE`, também no launcher) e o último controle usado |
| Touchpad sem touchpad | View/Select (ou Tab) + analógico direito desliza ("Open Door") |
| VRAM | orçamento padrão de 3500 MB (falta de VRAM no carregamento) |
| FPS caindo com o tempo | variantes de shader com a chave tirada antes da tradução: de 60→6 FPS para estável |
| Partículas da GPU (fumaça do dreno, drones) | leitura precisa da GPU por padrão; modo rápido no launcher ("Precisão da GPU") |
| Stencil `test_val`/`op_val` | uma referência Vulkan quando as máscaras permitem |
| Cache de shaders | pasta com a chave do tradutor; o cache antigo é apagado sozinho |
| Launcher Qt | padrão nos pacotes, 8 telas, pt-BR/en-US, Tk de reserva ("classic"), `--selftest` no CI |
| Idioma do jogo | "Automático (idioma do sistema)" por padrão e os 30 idiomas do PS4 que o dump marca (italiano, polonês, espanhol da Espanha... nos dumps europeus); sem `playgo-chunk.dat`, os 6 do CUSA00223 |
| Saves e dados | pasta configurável (`ISS_USER_DIR`, arquivo-ponteiro), achar saves de outras instalações, backup antes de cada partida, exportar/importar `.zip` |
| Correções experimentais | BP01–BP12 (shadPS4, SotC, bbport, Mrsuss/0xCydral, CTR) com chave individual, perfil Recomendado e `TLG_BACKPORTS=0`; log com `Config:`, `Backports:` e `Backport hits` |
| Oclusão | modo 3 (contagem zero) por padrão: corrige faróis que piscavam e brilho que atravessava personagens (Windows e Linux) |
| Mouse | câmera, botões e modo touchpad (`ISS_MOUSE_*`, `ISS_KEY_MAP`); falta validar em mais jogadores |
| Guias de controle | imagens Xbox/Switch Pro/8BitDo em pt-BR e en-US |
| Suporte e Atualizações | relatório sem caminhos pessoais; verificação da última release (sem instalar nada) |
| CPUs sem F16C | as conversões de meia precisão são emuladas na partida (#51) |

## Aberto (por prioridade)

| # | Item | Detalhe |
|---|---|---|
| 1 | Desempenho | a thread do jogo espera leituras da GPU (escritas de `memcpy` em páginas modificadas pela GPU); correção: rastrear a memória em pedaços menores que a página. O primeiro passo existe (`ISS_WRITE_MERGE`, ligado pelo launcher, e as leituras adiadas), e o FPS segue instável. Em 17/10: atalhos sem mudança de imagem (bits da GPU, `memset` mesclado, samplers do conjunto de texturas, escritas de descritores, apresentação) e, para teste A/B, `ISS_READBACK_LOCAL_JOIN=1`, `ISS_WRITE_MERGE_MIN` e `ISS_PHYS_CORES=1` ([parallel_gpu.md](../port/docs/parallel_gpu.md)) |
| 2 | Engasgo de shader | 64–118 compilações por janela de 5 s em cenas novas; o BP08 (blob do cache do driver) vem ligado desde 16/10 e a compilação síncrona não pula desenhos; o blob agora só é salvo com pipelines novos e tem limite de 1 GB (`ISS_DRIVER_CACHE_MAX_MB`); `ISS_PIPELINE_BINARIES=1` ainda derruba o jogo no driver NVIDIA 595 (agora desligado sem a variável). Em teste: `ISS_GFX_DEFERRED_BIND=1` compila o pipeline numa thread do compilador em vez da thread de comandos da GPU, sem pular desenhos |
| 3 | Oclusão real | o modo 3 é uma aproximação (contagem zero); falta medir a oclusão de verdade pelo Vulkan (occlusion queries). Risco atual: brilho de luzes mais fraco em algumas cenas (modo 0 volta ao antigo) |
| 4 | Névoa, água e fogo piscando; grade fina de linhas escuras | corrigido em 17/10, a confirmar no jogo: na redução de ballots wave64 em GPUs de 32 lanes, a metade ímpar de cada wave lia a máscara errada (a sua e a da wave seguinte), então as lanes 32–63 calculavam posições erradas nas listas de partículas e luzes; suspeitas: ballots wave64 em fluxo divergente, texturas BC 1D; os faróis e o brilho de luzes que piscavam já foram corrigidos (oclusão, modo 3). Candidata, a confirmar no jogo: em GPUs com subgrupo padrão de 32 e suporte a 64, os compute shaders rodavam com 64 lanes mas traduzidos para 32; agora são traduzidos para 64 (o cache de shaders é refeito uma vez depois de atualizar; o log mostra `GPU: host subgroup size`) |
| 5 | Brilho azul do dreno de drones | não aparece; ballot wave64, stencil ou readback |
| 6 | RX 580 / Polaris no Windows | crash em `amdvlk64.dll+0xebbbb` (#31, #49, #59, #65, #67, #81); causa em investigação; as BC 1D minúsculas são decodificadas na CPU (`ISS_BC_TINY`) e o BP09 está para teste |
| 7 | GPU de 4 GB de VRAM | em 17/10: nas GPUs integradas o orçamento fixo de 3500 MB do launcher passava por cima do orçamento atual do driver (agora vale o menor); a memória de textura estoura (#72, #75); a investigar; por enquanto baixe a resolução. "Precisão da GPU: Rápido" escurece a imagem |
| 8 | Quadro / Maxwell | pode estar abaixo do mínimo (#70); a investigar |
| 9 | Texturas BC 1D | promoção para 2D automática nos drivers que recusam BC 1D, menos a NVIDIA; `ISS_BLOCK1D=0/1` decide. Na NVIDIA fica desligada porque estraga o HUD |
| 10 | Atualizador automático | feito no launcher Qt (só pacote): "Baixar e instalar" pergunta antes de baixar, de instalar e de reiniciar, só em HTTPS, confere o tamanho e o SHA-256 (`SHA256SUMS`, publicado pelo CI) e instala numa pasta nova ao lado; a pasta antiga fica para voltar. Falta testar numa release real (Linux/Windows); o launcher Tk não tem atualizador |
| 11 | Mouse | validar com mais jogadores (câmera, botões, modo touchpad) |
| 12 | Windows | validar em mais PCs e placas (hoje o teste é por relatos de jogadores) |
| 13 | `param.sfo` 1.07 na visão do jogo | o `sce_sys` do update não traz o `param.sfo`, e a visão ficava com o 1.00 (sem ele os mods do 1.07 não aparecem). O `make_game_view.py` agora liga o de maior `APP_VER` (o do update, um `patch-meta/param.sfo` ou o passado como 4º argumento) e o launcher avisa quando o `param.sfo` não é o 1.07; falta conferir com um dump real |
| 14 | Idiomas | os launchers oferecem, entre os 30 idiomas do PS4, os que o `sce_sys/playgo-chunk.dat` do dump marca (máscara de idioma dos chunks do PlayGo; o `param.sfo` não diz isso), cada um com o próprio id (#194, #202, #162, #196); o "Automático" segue idioma e região do sistema e cai na variante mais próxima, depois no inglês; sem o arquivo (dump por FTP) ou sem chunks de idioma, ficam os 6 do CUSA00223. Falta conferir num dump real; o turco que não pega (#204) ainda não tem causa conhecida (o id 19 chega ao jogo) |
| 15 | Prefixo `ISS_`/`TLG_` | o jogador usa `ISS_*` e o núcleo ainda lê `TLG_*` por dentro; quando todos os ports usarem o ps4_core, o prefixo comum vira `PSCORE_` |

---

# inFAMOUS Second Son PC: roadmap (English)

Port of inFAMOUS Second Son (Sucker Punch, PS4, CUSA00223, base 1.00 + update 1.07, SDK 4.00).
Updated 15/10/2026. The player-facing status is in the [README](../README.md#english).

**Playable, in development:** boot, title, intro and gameplay (tested up to "Go to Longhouse"), own
audio (no firmware file), saves, correct HUD, real or simulated motion sensor, simulated touchpad and
GPU particles with precise readbacks (default). 20–30 FPS typical on an RTX 2060, unstable.

Done since the last roadmap: the Qt launcher as the default (Tk as a fallback), configurable saves and data,
12 experimental fixes (BP01–BP12), the occlusion counter fix (headlights and light glare; confirmed on Windows and
Linux), mouse camera and touchpad mode, controller guides, the Support and Updates screens, and the F16C emulation.

Open, by priority: (1) performance — the game thread waits on GPU readbacks (2026-10-17: image-neutral fast paths on by default; `ISS_READBACK_LOCAL_JOIN`, `ISS_WRITE_MERGE_MIN`, `ISS_PHYS_CORES` to A/B); (2) shader-compile stutter — `ISS_GFX_DEFERRED_BIND=1` builds missed pipelines on a compiler
thread instead of the GPU command thread (to A/B); (3) real occlusion through Vulkan (mode 3 is an approximation);
(4) flickering fog/water/fire and a fine dark grid (a candidate fix, to be confirmed in game: on GPUs with a default
subgroup size of 32 that support 64, compute shaders ran 64 lanes with the 32-lane wave64 lowering and are now lowered
for 64; the shader cache is rebuilt once after updating, and the log shows `GPU: host subgroup size`); (5) the missing blue drone-drain glow; (6) the RX 580/Polaris crash
on Windows; (7) GPUs with 4 GB of VRAM; (8) Quadro/Maxwell; (9) BC 1D textures; (10) an automatic updater (done in the Qt
launcher for packaged builds: download, SHA-256 check against the release's SHA256SUMS, install into a new
sibling folder; to be tried on a real release); (11) validating the mouse; (12) validating Windows on more PCs; (13) the 1.07 `param.sfo` in the
game view (make_game_view.py now links the newest one and the launcher warns about a 1.00 one; to be checked
with a real dump); (14) game languages from the game data (any of the PS4's 30 that the PlayGo chunks of `playgo-chunk.dat`
list, the 6 of CUSA00223 when it cannot tell; to be checked with a real dump; Turkish not applying, #204, is not
explained yet); (15) the `ISS_`/`TLG_`/`PSCORE_` prefix.
