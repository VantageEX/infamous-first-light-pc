<p align="center"><img src="port/packaging/icon/infamous-icon-1024.png" width="160" alt=""></p>

<h1 align="center">inFAMOUS Second Son — port nativo para PC</h1>

<p align="center"><a href="https://buymeacoffee.com/hayatog"><img src="https://img.shields.io/badge/Buy_me_a_coffee-Apoie_o_projeto-FFDD00?logo=buymeacoffee&amp;logoColor=000000" alt="Apoie com um café / Buy me a coffee"></a></p>

<p align="center">
  <a href="#english"><b>English below</b></a> ·
  <a href="https://github.com/HayatoG/iss_pc/releases/latest">Download</a> ·
  <a href="https://www.youtube.com/@hayatog-dev">YouTube</a> ·
  <a href="https://discord.gg/UMkdMYZh9g">Discord</a>
</p>

<p align="center"><a href="https://github.com/sponsors/HayatoG"><img src="https://img.shields.io/badge/Sponsor-HayatoG-ea4aaa?logo=githubsponsors&logoColor=white" alt="Sponsor HayatoG on GitHub"></a></p>

<p align="center"><a href="https://github.com/HayatoG/iss_pc/actions/workflows/build.yml"><img src="https://github.com/HayatoG/iss_pc/actions/workflows/build.yml/badge.svg" alt="Build"></a></p>

<p align="center">
  <a href="https://trendshift.io/repositories/307125?utm_source=trendshift-badge&amp;utm_medium=badge&amp;utm_campaign=badge-trendshift-307125" target="_blank" rel="noopener noreferrer"><img src="https://trendshift.io/api/badge/trendshift/repositories/307125/daily?language=C%2B%2B" alt="HayatoG/iss_pc | Trendshift (daily)" width="250" height="55"></a>
  <a href="https://trendshift.io/repositories/307125?utm_source=trendshift-badge&amp;utm_medium=badge&amp;utm_campaign=badge-trendshift-307125" target="_blank" rel="noopener noreferrer"><img src="https://trendshift.io/api/badge/trendshift/repositories/307125/weekly?language=C%2B%2B" alt="HayatoG/iss_pc | Trendshift (weekly)" width="250" height="55"></a>
</p>

> **Jogável, em desenvolvimento.** O jogo inicia, chega ao gameplay e passa pelos primeiros capítulos, mas o desempenho ainda é instável (20–30 FPS típicos numa RTX 2060) e há falhas de imagem conhecidas. Veja o [Status](#status).

**inFAMOUS Second Son** (PS4, CUSA00223, jogo base 1.00 + update 1.07) rodando no PC sem emulador de CPU: o código original do jogo roda nativo no processador, o sistema do PS4 é reimplementado e a GPU é traduzida para Vulkan. Usa o mesmo núcleo dos ports de [The Last Guardian](https://github.com/HayatoG/tlg_pc) e [Gravity Rush 2](https://github.com/HayatoG/gr2_pc).

> **Nenhum arquivo do jogo vem com este projeto.** Para jogar, você precisa da sua própria cópia, tirada do seu PS4.

Por **HayatoG** · YouTube [@hayatog-dev](https://www.youtube.com/@hayatog-dev) · Discord [discord.gg/UMkdMYZh9g](https://discord.gg/UMkdMYZh9g)

| | |
|---|---|
| ![Delsin diante do outdoor, com o pedido de virar o controle de lado](docs/screenshots/01-grafite-sensor.jpg) | ![Delsin no galpão do cais](docs/screenshots/02-galpao.jpg) |
| ![Na cerca do cais](docs/screenshots/03-cerca.jpg) | ![Correndo pela praia ao pôr do sol](docs/screenshots/04-praia.jpg) |

![Escalando as rochas acima da vila](docs/screenshots/05-escalada.jpg)

<sub>Capturas na resolução nativa (1920×1080, reduzidas para 1280), direto da GPU, no Linux com uma RTX 2060.</sub>

## Contribuidores / Contributors

<p align="center">
  <a href="https://github.com/HayatoG"><img src="https://avatars.githubusercontent.com/u/22659047?v=4" width="72" alt="Foto de perfil de contribuidor"></a>&nbsp;&nbsp;
  <a href="https://github.com/intemerated"><img src="https://avatars.githubusercontent.com/u/343755099?v=4" width="72" alt="Foto de perfil de contribuidor"></a>&nbsp;&nbsp;
  <a href="https://github.com/Andretuta"><img src="https://avatars.githubusercontent.com/u/219594641?v=4" width="72" alt="Foto de perfil de contribuidor"></a>&nbsp;&nbsp;
  <a href="https://github.com/Playdbunny"><img src="https://avatars.githubusercontent.com/u/70190491?v=4" width="72" alt="Foto de perfil de contribuidor"></a>&nbsp;&nbsp;
  <a href="https://github.com/EdouardSence"><img src="https://avatars.githubusercontent.com/u/103503538?v=4" width="72" alt="Foto de perfil de contribuidor"></a>&nbsp;&nbsp;
  <a href="https://github.com/HayatoG/iss_pc/graphs/contributors"><img src="https://github.com/identicons/guilherme-oliveira.png" width="72" alt="Avatar de contribuidor"></a>&nbsp;&nbsp;
  <a href="https://github.com/batuhankiskac"><img src="https://github.com/batuhankiskac.png" width="72" alt="Foto de perfil de contribuidor"></a>&nbsp;&nbsp;
  <a href="https://github.com/sapoguruoficial-cmd"><img src="https://github.com/sapoguruoficial-cmd.png" width="72" alt="Foto de perfil de contribuidor"></a>
</p>

## Novidades

- **Launcher novo (Qt Quick), agora o padrão:** com 10 telas, incluindo **Mods / Cheats** e **Avançado**, interface em português e inglês e o idioma do jogo seguindo o do sistema por padrão. O launcher antigo (Tk) continua no pacote como **inFAMOUS Second Son (classic)**, de reserva. Veja [O launcher](#o-launcher).
- **Saves e dados:** pasta de dados configurável, backup antes de cada partida, exportar/importar `.zip`, duplicar e excluir saves, e a busca de saves de outras instalações. Resolve o "perdi meu save depois de atualizar". Veja [Saves e dados](#saves-e-dados).
- **Faróis piscando e brilho que atravessava personagens: corrigido.** O contador de oclusão que o jogo lê para saber quanto de uma luz está visível era falso; o novo modo 3 (o padrão) devolve contagem zero, e isso corrigiu o problema, **confirmado no Windows e no Linux**. Veja [Luzes e oclusão](#luzes-e-oclusão).
- **Correções experimentais BP01–BP12:** doze correções trazidas de outros ports de PS4 e do shadPS4, com chave individual, perfil Recomendado e o "Como a release anterior" para comparar. Veja [Correções experimentais](#correções-experimentais).
- **Logs mais úteis:** todo log começa com um bloco `Config:` (suas opções, sem caminhos) e mostra as linhas `Backports:`; a tela **Suporte** monta o relatório sem dados pessoais. Veja [Logs e relatórios](#logs-e-relatórios).
- **Guias de controle:** imagens prontas (Xbox, Switch Pro e 8BitDo, em português e inglês) do modo sensor e do touchpad, na seção [Sensor de movimento e touchpad](#sensor-de-movimento-e-touchpad) e na tela **Guias de controle** do launcher.
- **Processadores sem F16C (Sandy Bridge, #51):** as 13 conversões de meia precisão do jogo (`VCVTPH2PS`/`VCVTPS2PH`) passam a ser emuladas na partida, com o mesmo resultado bit a bit do processador (arredondamento, denormais, Inf e NaN). Antes o jogo parava com "13 sites use encodings the rewrite cannot emulate".
- **Mouse:** o mouse gira a câmera (analógico direito) e também faz o papel do **touchpad**. Esquerdo = R2, direito = L2, do meio = R3. **Segure Ctrl esquerdo (ou o botão lateral X1)** e o mouse vira um dedo no touchpad: esquerdo toca/clica (segurar = absorver poder), arrastar com o esquerdo desliza (porta "Open Door", scanner), direito = dois dedos (levantar o carro do Reggie). A janela prende o mouse; Esc solta e um clique prende de novo. Veja [Mouse](#mouse).
- **Explicação do grafite na tela:** na primeira vez que o modo sensor liga, uma tela no meio mostra os passos (sacudir, pintar, mirar, sair) com os botões do seu controle; fecha com "Entendi". O menu Insert ganhou a aba **Controle**, que mostra essa tela de novo e ajusta a **velocidade da mira do grafite** (25–200%, também `ISS_SENSOR_AIM`).
- **Grafite mais fácil (modo sensor):** o modo sensor agora é a lata de spray do grafite: ao ligar (L3+R3), o controle já fica de lado, como o jogo pede ("Hold Wireless Controller Sideways"); **X/A sacode**, **R2 pinta** e o **analógico esquerdo mira** o cursor para os lados e para cima/baixo (antes ele só subia e descia), mais devagar com o analógico pouco inclinado, para contornar o desenho. No teclado, enquanto T ou G seguram o controle de lado, as **setas** miram.
- **Em teste: fumaça e partículas da GPU piscando (#7, #50).** Correções do tradutor de shaders, parte delas vinda do port nativo de [Shadow of the Colossus](https://github.com/AjncSystem/ShadowOfTheColossus-PC):
  - barreiras de memória compartilhada (LDS) nos shaders de compute de qualquer tamanho de workgroup, inclusive os que não são múltiplos de 64 (96, 192...): `ISS_LDS_SMALL` (menos de 64 threads), `ISS_LDS_MULTI` (mais de 64; em GPU de 32 lanes a barreira cobre o workgroup inteiro, porque cada wave de 64 do PS4 ocupa dois subgrupos), `ISS_LDS_READLANE`;
  - `ISS_LANE_DIVERGENCE`: desvios que dependem do `v_mbcnt`, do número da lane ou do valor devolvido por um atomic contam como divergentes, para nenhuma barreira cair dentro de um desvio que só uma lane toma;
  - `S_WQM_B64`, que não fazia nada (`ISS_WQM`).
  - Todas ligadas por padrão no inFAMOUS; `0` desliga cada uma, para comparar. O log mostra uma linha `shader switch` por chave.
  - Experimento, desligado: `ISS_RFL_ALL=1` troca o `ReadFirstLane` pela primeira lane da wave de 64 inteira em todos os shaders de compute largos (não só nos que usam ballot).
  - O cache de shaders se refaz sozinho a cada combinação.
- **Correções vindas do [shadPS4-Infamous-FixBuild](https://github.com/rayhlancaner-png/shadPS4-Infamous-FixBuild):**
  - **Brilho do Delsin ao drenar energia corrigido:** o stencil passa a respeitar o valor de teste e o de escrita, que são separados no PS4.
  - **Mais FPS:** quando o jogo lê algo da GPU, só a thread que pediu espera, não a GPU inteira (`ISS_DEFERRED_READBACK`, ligado por padrão), e o trabalho vai para a GPU em partes (`ISS_SUBMIT_DRAWS`, padrão 1024).
  - **Menos travadas ao surgir um efeito novo:** os shaders de compute compilam fora da thread da GPU (`ISS_COMPUTE_ASYNC_BUILD`), e o jogo tenta de novo quando o driver falha ao criar um pipeline.
  - **VRAM:** quando a memória de vídeo aperta, o port libera texturas antigas e tenta de novo, em vez de fechar. Uploads gigantes, como ao carregar um save, vão em partes, e o driver pode mover memória para a RAM.
  - **Atomics de float** corrigidos em placas sem suporte nativo, como as NVIDIA.
- **Sensor de movimento:** o giroscópio do DualShock 4/DualSense funciona direto. Em controles sem sensor, um toque em L3+R3 (ou Share no Xbox Series) liga o "modo sensor", a lata de spray do grafite: o controle fica de lado, X/A sacode, R2 pinta e o analógico esquerdo mira. No teclado, T/G viram de lado (com as setas mirando enquanto seguram) e Y sacode. É o que passa do grafite ("Hold Wireless Controller Sideways").
- **Analógicos:** zona morta de 10% contra drift (`ISS_STICK_DEADZONE`, também no launcher), e o último controle usado é o que vale.
- **Touchpad sem touchpad:** segure View/Select (ou Tab) e mova o analógico direito (ou as setas) para deslizar o dedo, como no "Open Door"; um toque rápido continua sendo clique.
- **Desempenho:** o FPS não cai mais com o tempo (caía de 60 para 6 em 2 minutos: as variantes de shader nunca se repetiam), e transições parciais de imagem repetidas são puladas.
- **Leitura precisa da GPU por padrão:** a fumaça do dreno de energia, os drones e outras partículas calculadas na GPU aparecem. O modo rápido continua no launcher ("Precisão da GPU").
- **Cache de shaders automático:** fica em `user/cache/`, junto dos saves (em Linux e Windows; o cache antigo de `~/.local/share/shadPS4` ou `user/user/` é trazido sozinho na primeira partida). A pasta leva a versão do tradutor de shaders, e o cache antigo é apagado sozinho depois de uma atualização. Para recompilar tudo, o launcher tem o botão **Limpar cache de shaders** (ou `./run_infamous.sh --clear-shader-cache`); ele apaga só os shaders compilados deste jogo, nunca saves ou configurações.

## Status

| Área | Estado |
|---|---|
| Boot, título, intro e gameplay | ✅ Funciona; testado até "Go to Longhouse", com escalada, corrida, poderes e cutscenes |
| Sensor de movimento (grafite, pedidos de movimento) | ✅ Giroscópio real ou simulado em qualquer controle e no teclado |
| Fumaça do dreno, drones e partículas da GPU | ✅ Com a leitura precisa da GPU (padrão) |
| HUD | ✅ Correto |
| Áudio | ✅ AudioOut + Ajm próprios, sem nenhum arquivo do firmware |
| Saves | ✅ O save é criado e o "Continue" funciona |
| Controle | ✅ DualSense, DualShock 4, Xbox e 8BitDo via SDL3 |
| Desempenho | 🔧 20–30 FPS típicos numa RTX 2060, instável; otimização em andamento |
| Engasgos | 🔧 Na primeira vez que cada shader é compilado (cutscenes e áreas novas) |
| Luzes (faróis piscando, brilho atravessando personagens) | ✅ Corrigido pelo modo de oclusão 3 (padrão), confirmado no Windows e no Linux; se o brilho das luzes ficar fraco, use o modo 0 |
| Imagem | 🔧 Alguns efeitos de névoa, água e fogo ainda piscam |
| Launcher | ✅ Qt Quick (padrão) com o Tk de reserva; português e inglês |
| Saves e dados | ✅ Pasta configurável, backup, exportar/importar |
| Mouse (câmera e touchpad) | 🧪 Implementado, ainda pouco testado em jogo |
| Correções experimentais (BP01–BP12) | 🧪 As de padrão ligado vêm no perfil Recomendado; BP07 e BP09 (desligadas) são experimentais; BP06 e BP08 ligadas por padrão |
| Linux | ✅ Testado (Nobara/Fedora 44) |
| Windows | 🧪 Testado por jogadores (a correção de luzes foi confirmada); crash conhecido em RX 580/Polaris, veja [Problemas conhecidos](#problemas-conhecidos) |

![Desempenho de uma sessão de 7 minutos andando pela cidade](docs/screenshots/desempenho.png)

<sub>FPS, pior quadro e draws por quadro a cada 5 s, tirados do log do port (`ISS_FRAME_STATS=1`), na RTX 2060: mediana de 27 FPS, 44% do tempo acima de 30 FPS. O jogo não tem trava de FPS. Gerado por `scripts/session_chart.py` (com `0` = sem trava).</sub>

## Download

A [**última versão**](https://github.com/HayatoG/iss_pc/releases/latest) é publicada quando o mantenedor decide reunir as mudanças em um pacote. PRs são compilados e testados; aprovações e merges não disparam builds. Na `main`, Linux e Windows só compilam na execução manual autorizada para release. As anteriores ficam em [Releases](https://github.com/HayatoG/iss_pc/releases) como histórico. Veja [como publicar](docs/RELEASES.md):

- **Linux x86-64** (glibc 2.39+: Ubuntu 24.04, Fedora 40, SteamOS 3.6 ou mais novos): `infamous-linux-x86_64.tar.gz`
- **Windows 10 (1903 ou mais novo) / 11, 64 bits**: `infamous-windows-x86_64.zip`

Os pacotes abrem o launcher Qt por padrão; ele soma cerca de 55 MB ao download (valor medido no Linux; no Windows é uma estimativa). Se o launcher novo não abrir na sua máquina, use o **inFAMOUS Second Son (classic)**, na mesma pasta.

## Arquivos do jogo necessários

Tudo vem do **seu** PS4 (GoldHEN + Itemzflow). Nada disso vai no download. Use **inFAMOUS Second Son CUSA00223 (EUA), jogo base 1.00 com o update 1.07**, montado numa pasta só, com o jogo base e o update por cima. São ~33 GB.

| Arquivo | Vem de | Para quê |
|---|---|---|
| `eboot.bin` | update 1.07 | o código do jogo (tem que ser o do 1.07) |
| `sce_module/libc.prx`, `sce_module/libSceFios2.prx` | update 1.07 | bibliotecas do próprio jogo |
| `sce_sys/` (`param.sfo`, `playgo-chunk.dat`, `icon0.png`, `pic1.png`) | jogo base (+ `param.sfo` do update) | metadados, instalação e a arte do launcher |
| `art/` (`cache/`, `movies/`, `ui/`) | base + update | conteúdo do jogo |
| `playgo.pgm`, `sce_discmap.plt`, `sce_discmap_patch.plt` | base + update | mapa de arquivos e instalação |

**Do firmware do console:** nada. O áudio usa o nosso AudioOut + Ajm; o jogo não importa `libSceNgs2` nem outro módulo do sistema do PS4.

A lista completa e o passo a passo, que também vão nas notas de cada Release, estão em [`docs/ARQUIVOS_NECESSARIOS.md`](docs/ARQUIVOS_NECESSARIOS.md).

**PC:** Windows 10 (1903+)/11 64 bits ou Linux x86-64 (glibc 2.39+), placa de vídeo com Vulkan 1.3 (6 GB de VRAM recomendados) e ~33 GB livres.

## Como jogar

1. Extraia o pacote.
2. Abra o **inFAMOUS Second Son** (no Windows, se aparecer "O Windows protegeu o computador", clique em *Mais informações* e *Executar assim mesmo*: o programa não é assinado digitalmente). Escolha a pasta do jogo e clique em **Jogar**. O idioma do jogo é o do seu sistema por padrão; dá para trocar na mesma tela. Se a imagem engasgar demais, a opção "Precisão da GPU" (tela **Gráficos**) tem um modo rápido (sem as partículas da GPU).
3. No jogo, **F10** liga o contador de FPS, e **Insert** (ou R3 + L2 no controle) abre o menu do port.
4. Quando o jogo pedir para virar ou sacudir o controle e o seu não tiver sensor: toque em L3+R3 (ou Share no Xbox Series). O controle já fica de lado; X/A sacode, R2 pinta e o analógico esquerdo mira. Toque de novo para sair. No teclado: T/G (com as setas para mirar) e Y. Veja [Sensor de movimento e touchpad](#sensor-de-movimento-e-touchpad).

O passo a passo completo está em [`README-player.md`](port/packaging/README-player.md), que também vem dentro do pacote.

## O launcher

O launcher é o programa que abre ao executar **inFAMOUS Second Son**. O novo (Qt Quick) é o padrão; o antigo (Tk) fica ao lado, como **inFAMOUS Second Son (classic)**, para o caso de o novo não abrir. Os dois usam as mesmas configurações e os mesmos saves, e aceitam `--play`, `--script` e `--clear-shader-cache`.

| Tela | Para que serve |
|---|---|
| **Jogar** | pasta do jogo, idioma do jogo, GPU e versão detectadas, perfil de correções, seletor de oclusão (luzes), botão Continuar/Jogar |
| **Gráficos** | modo de janela, VSync, limite de FPS, FSR 1 e nitidez, precisão da GPU, orçamento de VRAM, cache de shaders, overlay e predefinições (Desempenho, Equilibrado, Qualidade, Fiel ao PS4) |
| **Controles** | zona morta dos analógicos, velocidade da mira do grafite, mouse (câmera e modo touchpad), mapa do teclado e remapeamento |
| **Correções** | BP01–BP12, chaves de diagnóstico (A/B) e opções do menu Insert; perfil Recomendado e "Como a release anterior" |
| **Saves e dados** | pasta de dados, lista de saves, backups, exportar/importar `.zip`; veja [Saves e dados](#saves-e-dados) |
| **Guias de controle** | imagens Xbox, Switch Pro e 8BitDo (português e inglês): salvar, copiar, abrir a pasta |
| **Suporte** | problemas conhecidos para a sua GPU, links e o relatório de problema; veja [Logs e relatórios](#logs-e-relatórios) |
| **Mods / Cheats** | ativa cheats XML e o patch opcional para pular o vídeo inicial de logos (não pula cutscenes da história) |
| **Avançado** | configurações adicionais do launcher e do port |
| **Atualizações** | mostra a versão instalada e verifica a última release (veja abaixo) |

**Idioma da interface:** português (Brasil) ou inglês (EUA), que o launcher escolhe pelo idioma do sistema e dá para trocar. Quando você escolhe um, o launcher passa a escolha ao jogo (`ISS_UI_LANG`) para o overlay.

**Idioma do jogo:** a primeira opção é **Automático (idioma do sistema)**, o padrão, e depois vêm os idiomas que o `sce_sys/playgo-chunk.dat` do seu dump marca, entre os 30 do PS4 (inglês dos EUA e do Reino Unido, português do Brasil e de Portugal, espanhol latino e da Espanha, francês e francês do Canadá, alemão, italiano, holandês, polonês, turco, russo, japonês, coreano, chinês e outros); sem esse arquivo (dumps por FTP) vêm os 6 do CUSA00223: English (1), Português (Brasil) (17), Español (Latinoamérica) (20), Français (Canada) (22), Türkçe (19) e Русский (8). O automático segue o idioma e a região do sistema (es_ES vira espanhol da Espanha, es_MX vira latino, pt_PT vira português de Portugal) e, se o jogo não tiver esse idioma, usa a variante mais próxima (espanhol ↔ latino, português ↔ brasileiro...), senão o inglês. A escolha vale na próxima partida e vira `ISS_LANGUAGE` (um `ISS_LANGUAGE` definido por você vence). **Só funciona se os arquivos do seu jogo tiverem o idioma:** turco e russo, por exemplo, só aparecem se o seu dump tiver esses idiomas. É separado do idioma da interface.

**Atualizações:** a tela **Atualizações** faz uma única consulta HTTPS à última release no GitHub (nada seu é enviado), mostra as notas no idioma da interface e o botão **Baixar** só abre a página da release. **Ela não baixa nem instala nada: um atualizador automático de verdade ainda não existe.** "Verificar ao abrir" vem desligado.

**Rodar do código-fonte:** precisa de Python 3 com PySide6 (>= 6.7):

```bash
python3 -m venv .venv-qt
.venv-qt/bin/pip install -r port/launcher/qt/requirements.txt
./infamous_launcher_qt.sh      # o launcher antigo: ./infamous_launcher.sh
```

No Windows: `py -m venv .venv-qt`, `.venv-qt\Scripts\pip install -r port\launcher\qt\requirements.txt` e `.venv-qt\Scripts\python port\launcher\infamous_launcher_qt.py`. Sem o PySide6 o ponto de entrada diz como instalar. O pacote já leva tudo congelado, sem Python.

**Autoteste:** `"inFAMOUS Second Son" --selftest` abre o launcher fora da tela, confere QML, fontes, ícone e traduções, visita todas as telas e imprime `selftest ok` (não lê suas configurações nem saves e não inicia o jogo). O CI roda isso em cada pacote. Detalhes: [`docs/LAUNCHER_QT.md`](docs/LAUNCHER_QT.md) e [`docs/QT_PACKAGING.md`](docs/QT_PACKAGING.md).

## Correções experimentais

A tela **Correções** do launcher (e o botão **Correções experimentais…** do launcher Tk) liga e desliga, uma a uma, as 12 correções `BP01` a `BP12`, trazidas de outros ports de PS4 e do shadPS4. Cada uma tem a chave `ISS_BPnn` (ou `TLG_BPnn`) e a linha `bpnn` no `tlg.ini`; sem mexer em nada valem os padrões, que formam o perfil **Recomendado**. "No jogo" quer dizer que também muda pelo menu Insert, na hora; "reinicia" vale na próxima partida (as que mudam shader recompilam os shaders na primeira partida, e a imagem pode piscar ou ficar verde por um tempo).

| Id | O que faz | Origem | Padrão | Vale |
|---|---|---|---|---|
| BP01 | libera os buffers de comando aninhados | shadPS4 #5258 | ligado | no jogo |
| BP02 | busca de texturas mais rápida (só a primeira página) | shadPS4 #5240 | ligado | no jogo |
| BP03 | mantém o EXEC depois de um desvio divergente vazio | shadPS4 c34ca21f | ligado | reinicia, recompila shaders |
| BP04 | recíproco/raiz/log sem clamp, como o hardware | shadPS4 #5346 | ligado | reinicia, recompila shaders |
| BP05 | saturate trata NaN como a GPU (vira 0) | shadPS4 #5257 | ligado | reinicia, recompila shaders |
| BP06 | reconvergência máxima nos shaders de compute | port nativo de Shadow of the Colossus | **ligado** | reinicia, recompila shaders |
| BP07 | compilação assíncrona sem pular draws pequenos ou indiretos | bbport (bloodborne_pc) | **desligado** (experimental) | no jogo |
| BP08 | guarda o cache de pipelines do driver entre partidas | Mrsuss, 0xCydral (bloodborne_pc) | **ligado** (desde 16/10) | reinicia |
| BP09 | toda imagem 1D vira 2D de altura 1 | CTR | **desligado** (experimental) | reinicia, recompila shaders |
| BP10 | barreiras para buffers de vértice, índice e indiretos | shadPS4 #5262 | ligado | no jogo |
| BP11 | `firstInstance` contado uma vez nos draws indiretos | shadPS4 #5348 | ligado | reinicia, recompila shaders |
| BP12 | `EarlyFragmentTests` quando o jogo pede depth antes do shader | shadPS4 #5264 | ligado | reinicia, recompila shaders |

- **Como a release anterior:** `TLG_BACKPORTS=0` (ou o botão do launcher) desliga todas as BP que vêm ligadas, para comparar com a versão sem elas; um `TLG_BPnn` individual ainda vence. Na primeira abertura depois de atualizar, uma mensagem única explica isso. Se uma cena piorar, teste primeiro assim e depois ligue uma a uma.
- **Voltar ao recomendado** apaga todas as suas escolhas. Só o que difere do padrão é gravado.
- **Chaves de diagnóstico (A/B):** `ISS_WQM`, `ISS_LDS_SMALL`, `ISS_LDS_MULTI`, `ISS_LDS_READLANE`, `ISS_LANE_DIVERGENCE`, `ISS_RFL_ALL`, `ISS_BLOCK1D`, `ISS_BC_TINY`, `ISS_NO_MSAA`, `ISS_BACKGROUND_WARMUP`, `ISS_PIPELINE_BINARIES` (ainda derruba o jogo no driver NVIDIA 595) e o seletor de oclusão (`ISS_OCCLUSION_MODE`); `0` desliga, `1` liga. Servem para descobrir qual mudança causa um problema.
- O BP08 (cache de pipelines do driver em disco) vem ligado desde 16/10: a criação de pipelines fica rápida o bastante para nenhum desenho ser pulado (os flashes cinza de um quadro). O arquivo é salvo só quando há pipelines novos e não passa de 1 GB (`ISS_DRIVER_CACHE_MAX_MB`; apagar o arquivo recomeça). O resto do engasgo de shader está no [roadmap](docs/ROADMAP.md).
- Tabela completa, com as chaves do `tlg.ini` e as opções do menu Insert: [`docs/EXPERIMENTAL_FIXES.md`](docs/EXPERIMENTAL_FIXES.md).

## Luzes e oclusão

Os faróis piscavam e o brilho das luzes aparecia **através** dos personagens. A causa: o contador de oclusão (`ZPASS_DONE`) que o jogo lê para saber quanto de cada luz está visível era um número falso, que só crescia. Com o modo **3** (contagem zero, o padrão desta versão) o problema **foi corrigido, confirmado no Windows e no Linux**.

| `ISS_OCCLUSION_MODE` | Efeito |
|---|---|
| `0` | comportamento antigo (contador falso; o brilho pode atravessar) |
| `1` | teste: as luzes contam como tapadas |
| `2` | teste: as luzes contam como totalmente visíveis |
| `3` | contagem zero: **padrão** |

O seletor **Brilho das luzes (oclusão)** está na tela **Jogar** do launcher (e entre as chaves de diagnóstico); a variável também aceita o prefixo `TLG_`. **Risco:** em algumas cenas o brilho das luzes pode ficar mais fraco com o modo 3; se isso incomodar, use o modo 0. Esta é uma solução por aproximação; a oclusão real, medida pelo Vulkan, ainda não existe (veja o [roadmap](docs/ROADMAP.md)). Se você achar uma cena pior, mande o log e uma captura de cada modo.

## Mouse

O mouse funciona com ou sem controle (os botões dele somam aos do controle). A janela do jogo prende o mouse enquanto tem o foco e o menu Insert está fechado; **Esc solta** (o cursor volta) e **um clique na janela prende de novo** (esse clique não vira botão do jogo).

**Câmera e botões:**

| Mouse | No jogo |
|---|---|
| Mover | gira a câmera (analógico direito) |
| Botão esquerdo | R2 (usar o poder / atirar) |
| Botão direito | L2 (mirar) |
| Botão do meio | R3 |
| Botões laterais X1 / X2 e rodinha | livres (veja `ISS_KEY_MAP`); X1 também é a tecla do modo touchpad |
| Sensor ligado (modo sensor, T ou G) | o mouse aponta a lata de spray do grafite em vez de girar a câmera |

**Modo touchpad:** segure **Ctrl esquerdo** ou o **botão lateral X1** do mouse. Enquanto o modo está ligado a câmera não gira: o movimento do mouse leva um dedo virtual pelo touchpad (um quadradinho no canto inferior esquerdo da tela mostra onde ele está) e os botões viram o dedo:

| Gesto (mouse no modo touchpad) | O que o jogo vê | Onde usar |
|---|---|---|
| Só mover | dedo suspenso (nenhum toque): posiciona o dedo | antes de um gesto |
| Esquerdo, tocar rápido | toque + clique do touchpad | interações "aperte o touchpad" |
| Esquerdo, segurar parado | toque + clique mantidos | absorver/drenar poderes (fumaça, neon...) |
| Esquerdo, arrastar mais de 80 unidades | o clique solta e o dedo desliza | porta "Open Door", scanner de mão |
| Botão do meio (segurar e mover) | dedo sem clique | deslizar sem clicar, se um gesto não gostar do clique |
| Direito | dois dedos nas duas pontas com o touchpad apertado | levantar o carro de cima do Reggie |

O dedo começa no meio do touchpad e fica onde você o deixou. Os atalhos do teclado (Tab/Backspace, Tab + setas) continuam como antes.

**Variáveis de ambiente** (`ISS_` e `TLG_` valem igual; `mouse_*` também podem ir no `tlg.ini`):

| Variável | Padrão | O que faz |
|---|---|---|
| `ISS_MOUSE` | `1` | `0` desliga o mouse (a janela não prende mais) |
| `ISS_MOUSE_SENS` | `1.0` | sensibilidade da câmera (0,05–10) |
| `ISS_MOUSE_INVERT_Y` | `0` | `1` inverte o eixo Y |
| `ISS_MOUSE_SMOOTHING` | `0.5` | suavização (0–0,95) |
| `ISS_MOUSE_ACCEL` | `0` | aceleração (0–2) |
| `ISS_MOUSE_DEADZONE` | `0.1` | anti-zona morta: movimentos pequenos pulam a zona morta do jogo (0–0,5) |
| `ISS_MOUSE_PAD` | `1` | `0` desliga o modo touchpad do mouse |
| `ISS_MOUSE_PAD_KEY` | `Left Ctrl\|Mouse X1` | tecla(s) do modo touchpad; nomes do SDL ou `Mouse Left/Middle/Right/X1/X2`, separados por `\|` |
| `ISS_MOUSE_PAD_MODE` | `hold` | `toggle`: um toque na tecla liga, outro desliga |
| `ISS_MOUSE_PAD_SENS` | `1.0` | velocidade do dedo (0,1–10) |
| `ISS_MOUSE_PAD_DRAG` | `80` | unidades (touchpad de 1920 de largura) de arrasto até o clique virar deslize; `0` = nunca |
| `ISS_KEY_MAP` | — | remapeia teclas e mouse: `"r2=Mouse Left\|F,l2=Mouse Right\|R,r1=Mouse X2,l1=Wheel Down"`; vários por botão com `\|`, `none` tira todos |
| `ISS_MOUSE_LOG` | `0` | `1` mostra no log o que o jogo recebe do mouse |

O menu Insert, aba **Controle**, tem os ajustes da câmera do mouse, e a tela **Controles** do launcher tem os mesmos ajustes, o modo touchpad e o mapa do teclado. **O mouse é novo e ainda foi pouco testado em jogo por outros jogadores**: se algo se comportar mal, `ISS_MOUSE=0` o desliga e o relato ajuda.

**Teclado (mapa padrão):** funciona com ou sem controle (as teclas somam aos botões do controle).

| Tecla | Botão do PS4 | Tecla | Botão do PS4 |
|---|---|---|---|
| W / S (A / D) | analógico esquerdo | setas | analógico direito |
| Espaço | X | Shift esquerdo | O (círculo) |
| E | quadrado | Q | triângulo |
| 1 | L1 | 3 | R1 |
| R | L2 | F | R2 |
| Z | L3 | C | R3 |
| Enter | Options | Tab / Backspace | clique do touchpad (esquerdo / direito) |
| I / K / J / L | direcional (cima / baixo / esquerda / direita) | T / G / Y | modo sensor simulado (veja abaixo) |

**Remapear:** `ISS_KEY_MAP` (ou o campo avançado da tela **Controles**) troca teclas e botões do mouse: `"cross=Q,r2=Mouse Left|F"`. Os botões são `cross`, `circle`, `square`, `triangle`, `l1`, `r1`, `l2`, `r2`, `l3`, `r3`, `options`, `up`, `down`, `left`, `right`, `touchpad_left`, `touchpad_right`.

## Saves e dados

Por padrão os saves e as configurações ficam em `user/`, ao lado do jogo. **A atualização não apaga nada** (o zip não traz `user/`), mas se você extrair o zip novo em **outra pasta**, ou apagar a pasta antiga, o jogo abre sem saves: eles continuam na pasta antiga, em `user/savedata`. É por isso que parece que "a atualização apagou o save".

**Receita para atualizar sem perder saves:**
1. Abra o launcher, vá em **Saves e dados** e clique em **Fazer backup agora** (e, se quiser, **exportar tudo** para um `.zip`).
2. Escolha uma pasta de dados fixa fora do jogo (sugestão: `Documentos\inFAMOUS-PC` no Windows, `~/.local/share/infamous-pc` no Linux). O launcher pergunta se copia seus saves e configurações para lá (copia, nunca move).
3. Baixe e extraia a versão nova onde quiser: ela encontra os saves pela escolha, que fica num arquivo do seu usuário e sobrevive a reinstalações.
4. Se você já extraiu a versão nova e o save sumiu, a tela acha as outras instalações (Downloads, Área de Trabalho, Documentos, pastas vizinhas...) e oferece **Importar**.

**Onde fica a pasta de dados**, nesta ordem: a variável `ISS_USER_DIR`; o arquivo-ponteiro do usuário (`%APPDATA%\inFAMOUS-PC\launcher.json` no Windows, `~/.config/infamous-pc/launcher.json` no Linux); e, sem nenhum dos dois, `user/` ao lado do jogo (modo portátil). A mesma pasta guarda `tlg.ini`, saves, backups e o cache de shaders.

**Na tela Saves e dados:** um cartão por slot (título, capítulo, karma, data, tamanho e a miniatura do próprio save) com exportar `.zip`, duplicar e excluir; importar `.zip`; backups com **Restaurar**, manter os últimos N (padrão 10) e uma pasta extra de backup (outro disco). **Um backup é feito antes de cada partida** (pulado se nada mudou). Toda operação que escreve num save exige o jogo fechado e faz um backup de segurança antes; um `.zip` inválido ou perigoso (caminhos com `..`, absolutos, links) é recusado sem escrever nada.

## Sensor de movimento e touchpad

O jogo pede o sensor de movimento e o touchpad em várias cenas: grafite, absorver poderes (fumaça, neon...), abrir portas e outras interações. No **DualShock 4 e no DualSense** o giroscópio e o touchpad funcionam direto, sem ligar nada. Nos outros controles e no teclado, o port simula os dois.

**Guias de controle** (as mesmas imagens estão na tela **Guias de controle** do launcher, com botões para salvar e copiar):

| Xbox / Xbox Series | Nintendo Switch Pro | 8BitDo |
|---|---|---|
| ![Guia para Xbox, em português](port/launcher/qt/assets/controller_guides/inFAMOUS-controle-xbox-pt-BR.png) | ![Guia para Switch Pro, em português](port/launcher/qt/assets/controller_guides/inFAMOUS-controle-switch-pt-BR.png) | ![Guia para 8BitDo, em português](port/launcher/qt/assets/controller_guides/inFAMOUS-controle-8bitdo-pt-BR.png) |

**Modo sensor (ligar e desligar):**
- **Um toque em L3+R3 juntos** liga o modo sensor; outro toque desliga. Funciona em qualquer controle. Também serve o botão extra do controle: **Share** no Xbox Series, **Capture** no Switch Pro e **microfone** no DualSense. No DualShock 4 e no DualSense, o Share/Create **não** liga o modo sensor: ele é o clique do touchpad.
- Um aviso na tela confirma: "Modo sensor LIGADO (lata de spray): X sacode, R2 pinta, analógico esquerdo mira" ou "Modo sensor desligado".
- Com o modo ligado, o controle fica **de lado**, como uma lata de spray (o que o grafite pede), **X/A sacode** e o **analógico esquerdo mira**, girando a lata para os lados e para cima/baixo (é o que move o cursor do grafite; ele fica onde você soltou; pouco inclinado vai devagar, para contornar o desenho; o direito também serve). R2 continua pintando. O analógico e o X/A ficam escondidos do jogo, então o Delsin não anda, não pula e a câmera não gira: **desligue o modo depois da cena**. O L3+R3 juntos também não chega ao jogo.
- Com o modo ligado, a inclinação simulada substitui o giroscópio real.
- **Teclado** (sem modo, a qualquer hora): segure **T** ou **G** para virar o controle de lado (um lado ou o outro) e **Y** para sacudir.

**Touchpad em controle sem touchpad:**
- **View/Select/Back** é o clique do touchpad (lado direito; com L1 segurado, o lado esquerdo). Segurar o botão é segurar o touchpad.
- **Segure View/Select e mova o analógico direito** para deslizar o dedo a partir do meio do touchpad. Um toque rápido sem mover continua sendo clique.
- **Teclado:** Tab (lado esquerdo) e Backspace (lado direito) clicam; **segure Tab e use as setas** para deslizar.
- **Dois dedos nas duas pontas do touchpad** (levantar o carro de cima do Reggie): **segure View/Select + R1**; no teclado, **segure Tab + Backspace**.

| Ação no jogo | DualShock 4 / DualSense | Xbox / outro controle | Teclado |
|---|---|---|---|
| Absorver/drenar um poder (fumaça, neon...) e interações de "segure o touchpad" | segurar o touchpad | segurar View/Select (sem mexer o analógico direito, senão vira deslize) | segurar Tab ou Backspace (sem as setas); no mouse: Ctrl esquerdo (ou X1) + segurar o esquerdo parado |
| Interação de "toque no touchpad" | clicar o touchpad | toque em View/Select | Tab ou Backspace |
| Grafite: virar o controle de lado | virar o controle | L3+R3 (modo sensor): já fica de lado | segurar T ou G |
| Grafite: sacudir a lata | sacudir o controle | X/A no modo sensor | Y |
| Grafite: pintar | R2, mirando com o controle | R2, mirando com o analógico esquerdo no modo sensor | F (R2), mirando com as setas enquanto segura T ou G |
| Apertar as duas pontas do touchpad (levantar o carro de cima do Reggie) | dois dedos, um em cada ponta | segurar View/Select + R1 | segurar Tab + Backspace; no mouse: Ctrl esquerdo (ou X1) + botão direito |
| Deslizar no touchpad (porta "Open Door", scanner de mão) | deslizar o dedo | segurar View/Select + analógico direito na direção pedida | segurar Tab + setas; no mouse: segurar Ctrl esquerdo (ou X1), esquerdo e arrastar |
| Voltar ao normal depois do grafite | — | L3+R3 de novo ("Modo sensor desligado") | — |

**Launcher:** o quadro **Controle** tem a **Zona morta dos analógicos** (0–40%, padrão 10%). Aumente se o personagem, a câmera ou a inclinação do modo sensor se mexerem sozinhos (drift).

**Variáveis de ambiente:** `ISS_MOTION_BUTTON=<botão do SDL>` troca o botão do modo sensor (padrão `misc1`; o L3+R3 vale sempre), e `ISS_MOTION=off` ignora o giroscópio real (a simulação continua funcionando).

**Processadores Intel:** o código do jogo usa instruções SSE4A (EXTRQ), que só existem em processadores AMD. Num Intel o port troca essas instruções por uma emulação na partida (precisa do `iced-x86`, que já vem nos pacotes). Processadores mais antigos (Sandy e Ivy Bridge) também não têm MOVBE, F16C, BMI1 ou LZCNT, que são emulados do mesmo jeito; o AVX continua obrigatório. `ISS_ISA_REWRITE=sse4a` (ou `f16c`, ou uma lista como `f16c,movbe`) força a emulação num processador que tem a instrução (teste), `ISS_ISA_REWRITE=1` força todos os conjuntos emulados (MOVBE, BMI1, LZCNT, F16C, SSE4A) e `ISS_ISA_REWRITE=0` desliga. Detalhes: [port/docs/isa_rewrite.md](port/docs/isa_rewrite.md).

**Windows: proteção da red zone.** O código do PS4 guarda variáveis nos 128 bytes abaixo do `rsp` (a "red zone"), e o Windows escreve ali a cada exceção (o port usa exceções para acompanhar a memória da GPU). Na partida o port reescreve as instruções de memória das funções que usam a red zone para rodarem com o `rsp` 128 bytes mais baixo (crash em `0x1cfaaf`, #29/#41). Vem ligada no Windows; `ISS_RED_ZONE_REWRITE=0` desliga e `ISS_RED_ZONE_REWRITE=1` liga no Linux (teste). Teste no Linux: `TLG_CLOBBER_RED_ZONE=<bytes>` apaga essa quantidade de bytes abaixo do `rsp` a cada exceção, como o Windows (`1` = 128). Detalhes: [port/docs/red_zone.md](port/docs/red_zone.md).

**Tela congelada:** se o jogo passa 10 s sem mostrar um quadro novo, o log (`out/last_run.log`) ganha um bloco `FREEZE: no frame for 10 s; threads:` com cada thread do jogo, o que ela está esperando (mutex, semáforo, fila de eventos, leitura de arquivo, GPU...) e onde está no código. O jogo continua rodando, e o relatório só se repete depois que os quadros voltarem e pararem de novo. Mande esse log junto com o relato. `ISS_FREEZE_REPORT_SECONDS=<s>` muda o tempo (`0` desliga).

**Placas AMD no Windows (RX 580): compatibilidade experimental.** Há relatos de crash depois dos logos em `amdvlk64.dll+0xebbbb` (#31, #49). A investigação do [PR #232](https://github.com/HayatoG/iss_pc/pull/232) encontrou um lote de 9.217 barreiras, incluindo 9.216 para uma única imagem de 1.536 camadas e 6 mips; o log não prova que a chamada foi a causa do crash. Dois jogadores relataram que um build de teste passou desse ponto. Em **Gráficos → Imagem**, ligue **Compatibilidade RX580**, ou escolha **Compatibilidade RX 580 (experimental)** para aplicar também as escolhas recomendadas para AMD. O teste fica desligado por padrão e exige reiniciar o jogo; ainda precisa de validação nesta versão. A opção também aparece em **Avançado → Contornos de imagem**. Pelo terminal: `ISS_BARRIER_MERGE=1` liga a junção de barreiras vizinhas equivalentes; `0` desliga. Valores definidos fora do launcher têm prioridade. O contorno anterior para texturas BC minúsculas continua disponível (`ISS_BC_TINY=0/1`), mas não é uma correção confirmada desse crash. Para comparação de diagnóstico, **Avançado** expõe `ISS_EXPLICIT_RANGE` (padrão `1`, contagens explícitas de mips/camadas) e `ISS_BARRIER_SPLIT` (padrão `0`; `1` divide por mip, `2` por mip e camada; exige intervalos explícitos). Essas experiências isoladas não resolveram o crash nos testes do autor. Crédito: @intemerated pela junção e investigação; @Andretuta pela integração inicial no launcher.

## Logs e relatórios

Cada partida grava um log em `out/logs/run-AAAAMMDD-HHMMSS.log` (os mais recentes ficam; `out/last_run.log` aponta para o último). O log começa com a linha `Build:` e um bloco **`Config:`** com as chaves do seu `tlg.ini` e as variáveis `TLG_`/`ISS_` definidas (valores que parecem caminhos ficam de fora), e traz as linhas `Backports:` (cada BP com o estado ligado/desligado e o modo de oclusão) e, junto das estatísticas de quadros, `Backport hits (total): ...` (quantas vezes cada correção realmente rodou). Com isso um relato diz quais correções estavam ligadas.

**Como reportar um problema:** abra a tela **Suporte** do launcher e use **Relatar um problema**: ela coleta GPU, driver, VRAM, sistema, CPU, versão do port e do jogo, deixa escolher o log, **tira dos textos os caminhos pessoais** (nome de usuário e pasta) e oferece **Copiar relatório**, **Salvar .zip** (relatório e log sem caminhos) e **Abrir issue no GitHub**. Nada é enviado sem o seu clique. Para reportar à mão, anexe o log em uma [issue](https://github.com/HayatoG/iss_pc/issues) ou mande no [Discord](https://discord.gg/UMkdMYZh9g).

**Log grande:** o GitHub recusa arquivos grandes. Compacte o log em `.zip` (a tela Suporte já faz isso) antes de anexar.

## Problemas conhecidos

A tela **Suporte** do launcher mostra os que casam com a sua GPU (uma tabela local, sem consulta à internet). Os conhecidos hoje:

| Situação | O que acontece | O que fazer |
|---|---|---|
| **AMD RX 580 / Polaris no Windows** (#31, #49, #59, #65, #67, #81) | crash logo depois dos logos em `amdvlk64.dll+0xebbbb` | **correção experimental disponível.** Teste o perfil **Compatibilidade RX 580 (experimental)** no launcher, reinicie o jogo e mande o log; ainda sem garantia de correção |
| **GPU com 4 GB de VRAM** (#72, #75) | a memória de textura estoura; FPS muito baixo | baixe a resolução (tela Gráficos) e mantenha o orçamento de VRAM abaixo da memória da placa. Atenção: **"Precisão da GPU: Rápido" escurece a imagem** |
| **NVIDIA Quadro / Maxwell** (#70) | pode estar abaixo do mínimo do port; tela preta ou jogo que não abre | conte o que você viu; ainda sem solução |
| **CPU sem SSE4A ou F16C** (Intel, Sandy Bridge...) | o port emula essas instruções na partida | é automático; veja [Processadores Intel](#sensor-de-movimento-e-touchpad) |
| **Versão do jogo** | testado com **CUSA00223 1.00 + 1.07**; o CUSA00004 também inicia | use o CUSA00223 com o update 1.07 |

Placas NVIDIA RTX 30 no Windows já relataram luzes e fumaça piscando; use o perfil Recomendado e jogue duas partidas seguidas (a primeira recompila os shaders). Para os demais problemas abertos, veja as [issues](https://github.com/HayatoG/iss_pc/issues).

## Compilar do código-fonte

```bash
git clone --recursive https://github.com/HayatoG/iss_pc.git && cd iss_pc
cd port && ISS_LTO=OFF bash build.sh   # Linux: SDL3, Vulkan, FFmpeg, FreeType, Boost, fmt, Zydis, glslang
cd .. && ./infamous_launcher_qt.sh     # launcher Qt (precisa do PySide6); ./infamous_launcher.sh é o Tk; ./run_infamous.sh pelo terminal
```

Um clone compila sem otimização guiada por perfil (PGO): o perfil (`port/pgo/*.gcda`) não está no repositório e os pacotes das releases também são compilados sem ele, então um build local e um pacote devem render igual. `ISS_PGO=generate` gera um binário instrumentado que grava o perfil enquanto você joga, e o build seguinte o usa.

No Windows, o build roda no MSYS2 CLANG64 ([`port/packaging/windows/README.md`](port/packaging/windows/README.md)). O workflow [`.github/workflows/build.yml`](.github/workflows/build.yml) mostra o processo completo nas duas plataformas.

## Como funciona

| Camada | O que faz | Base |
|---|---|---|
| Loader | Carrega o `eboot.bin` 1.07 e liga os módulos do próprio jogo (`libc`, `Fios2`) | [tlg_pc](https://github.com/HayatoG/tlg_pc), [gr2_pc](https://github.com/HayatoG/gr2_pc), [gow3_pc](https://github.com/FelipePrado0/gow3_pc), [bloodborne_pc](https://github.com/deadinside28/bloodborne_pc) |
| Runtime | Reimplementa as bibliotecas do PS4 (NpManager, Trophy, Http, PlayGo, SaveData, AppContent, AudioOut, Ajm...). Específicos do inFAMOUS: `port/src/runtime_infamous.c` | `port/src/` |
| Vídeo | Traduz os comandos Gnm e os shaders GCN para Vulkan/SPIR-V | núcleo de vídeo do [shadPS4](https://github.com/shadps4-emu/shadPS4) |

## Próximos passos

O que falta está em [`docs/ROADMAP.md`](docs/ROADMAP.md): desempenho (esperas por leituras da GPU), cache de pipelines em disco contra os engasgos (o BP08 é só o primeiro passo), oclusão real pelo Vulkan, os efeitos de névoa, água e fogo que ainda piscam, o brilho do dreno de drones, o crash da RX 580 no Windows, um atualizador automático e a validação do mouse.

## Créditos e aviso legal

Base de código: [tlg_pc](https://github.com/HayatoG/tlg_pc), [gr2_pc](https://github.com/HayatoG/gr2_pc), [gow3_pc](https://github.com/FelipePrado0/gow3_pc) (FelipePrado0), [bloodborne_pc](https://github.com/deadinside28/bloodborne_pc) e [shadPS4](https://github.com/shadps4-emu/shadPS4) o fork [shadPS4-Infamous-FixBuild](https://github.com/rayhlancaner-png/shadPS4-Infamous-FixBuild) (Eye Of The Ruins/rayhlancaner-png, TOXIC-KID, 0xDeftones) e o port nativo [ShadowOfTheColossus-PC](https://github.com/AjncSystem/ShadowOfTheColossus-PC) (AjncSystem; com partes do Pink-shadPS4 de luizgustavs), todos GPL-2.0, assim como este projeto ([LICENSE](port/LICENSE)). Áudio ATRAC9 via [LibAtrac9](https://github.com/Thealexbarney/LibAtrac9); fontes via [FreeType](https://freetype.org).

**Origem das correções experimentais (BP01–BP12):** as correções do tradutor de shaders e do vídeo vêm do [shadPS4](https://github.com/shadps4-emu/shadPS4) (BP01–BP05, BP10–BP12), do port nativo de [Shadow of the Colossus](https://github.com/AjncSystem/ShadowOfTheColossus-PC) (BP06), do bbport, o [bloodborne_pc](https://github.com/deadinside28/bloodborne_pc) (BP07), do [bloodborne_pc_windows_port](https://github.com/Mrsuss60/bloodborne_pc_windows_port) de Mrsuss e do [bloodborne_pc](https://github.com/0xCydral/bloodborne_pc) de 0xCydral (BP08) e do CTR ([T3CHNOLOG1C/CTR](https://github.com/T3CHNOLOG1C/CTR)) (BP09). Os commits exatos estão em [`docs/EXPERIMENTAL_FIXES.md`](docs/EXPERIMENTAL_FIXES.md) e em `port/gpu/VENDOR.txt`. Fontes do launcher: Big Shoulders Stencil Display e Barlow (licença OFL, em `port/launcher/qt/fonts/`).

### Agradecimentos

- **Kazu**, pelos logs, capturas de tela e testes, no Windows e no Linux, que levaram à correção dos faróis que piscavam e do brilho que atravessava os personagens (o modo de oclusão 3).
- Aos jogadores que reportaram problemas nas [issues](https://github.com/HayatoG/iss_pc/issues), com logs que mudaram o rumo do port: jacckkss (#50, luzes e fumaça piscando), heyzerr (#51, #65, #67, CPUs sem F16C e RX 580), Andrade2244 (#29), MrRobot11214 (#31), fghacksr (#49), AngelHomura (#59) e aledieb93-svg (#81) (crashes na RX 580 e no Windows), ViniSnapeS (#72) e kudetgaming-art (#75) (memória de vídeo), summer882 (#71), manymanyyyd-afk (#63) e Kpopping (#73) (idiomas turco e russo), mahmoudmahran2112-crypto (#70, Quadro/Maxwell), Milicksp (#16), gsev3333 (#41) e todos os outros que abriram uma issue ou testaram uma versão.

Este projeto não tem vínculo com a Sony Interactive Entertainment ou a Sucker Punch Productions. inFAMOUS é marca da Sony Interactive Entertainment. Nenhum código, arquivo ou arte do jogo é distribuído aqui. O ícone é uma arte original do projeto.

---

<a id="english"></a>

# inFAMOUS Second Son — native PC port

> **Playable, in development.** The game boots, reaches gameplay and gets through the first chapters, but performance is still unstable (20–30 FPS typical on an RTX 2060) and there are known graphics glitches. See the status below.

**inFAMOUS Second Son** (PS4, CUSA00223, base game 1.00 + update 1.07) running on PC without a CPU emulator: the game's original code runs natively on the processor, the PS4 system is reimplemented, and the GPU is translated to Vulkan. It uses the same core as the [The Last Guardian](https://github.com/HayatoG/tlg_pc) and [Gravity Rush 2](https://github.com/HayatoG/gr2_pc) ports.

> **No game files come with this project.** To play, you need your own copy, dumped from your own PS4.

By **HayatoG** · YouTube [@hayatog-dev](https://www.youtube.com/@hayatog-dev) · Discord [discord.gg/UMkdMYZh9g](https://discord.gg/UMkdMYZh9g)

The screenshots are at the top of this page.

## What's new

- **New launcher (Qt Quick), now the default:** 10 screens, including **Mods / Cheats** and **Advanced**, the interface in Portuguese and English, and the game language following the system's by default. The old launcher (Tk) stays in the package as **inFAMOUS Second Son (classic)**, as a fallback. See [The launcher](#the-launcher).
- **Saves and data:** a configurable data folder, a backup before every play session, `.zip` export/import, duplicate and delete saves, and a search for saves from other installs. It fixes the "I lost my save after updating" problem. See [Saves and data](#saves-and-data).
- **Flickering headlights and glare that showed through characters: fixed.** The occlusion counter the game reads to know how much of a light is visible was a fake; the new mode 3 (the default) answers with zero counts, and that fixed the problem, **confirmed on Windows and on Linux**. See [Lights and occlusion](#lights-and-occlusion).
- **Experimental fixes BP01–BP12:** twelve fixes brought from other PS4 ports and from shadPS4, each with its own switch, a Recommended profile and a "Like the previous release" button to compare. See [Experimental fixes](#experimental-fixes).
- **More useful logs:** every log starts with a `Config:` block (your options, no paths) and shows the `Backports:` lines; the **Support** screen builds the report without personal data. See [Logs and reports](#logs-and-reports).
- **Controller guides:** ready-made images (Xbox, Switch Pro and 8BitDo, in Portuguese and English) of sensor mode and the touchpad, in [Motion sensor and touchpad](#motion-sensor-and-touchpad) and in the launcher's **Controller guides** screen.
- **CPUs without F16C (Sandy Bridge, #51):** the game's 13 half-precision conversions (`VCVTPH2PS`/`VCVTPS2PH`) are now emulated at start, bit-exact with the CPU (rounding, denormals, Inf and NaN). The game used to stop with "13 sites use encodings the rewrite cannot emulate".
- **Mouse:** the mouse turns the camera (right stick) and also plays the **touchpad**. Left = R2, right = L2, middle = R3. **Hold Left Ctrl (or the X1 side button)** and the mouse becomes a finger on the touchpad: left touches/clicks (hold = absorb a power), dragging with the left slides (the "Open Door" swipe, the scanner), right = two fingers (lift the car off Reggie). The window captures the mouse; Esc releases it and a click captures it again. See [Mouse](#mouse-1).
- **Graffiti steps on screen:** the first time sensor mode turns on, a screen in the middle shows the steps (shake, spray, aim, leave) with your controller's buttons; "Got it" closes it. The Insert menu has a new **Controls** tab that shows it again and sets the **graffiti aiming speed** (25–200%, also `ISS_SENSOR_AIM`).
- **Easier graffiti (sensor mode):** sensor mode is now the graffiti's spray can: turning it on (L3+R3) holds the controller on its side, as the game asks ("Hold Wireless Controller Sideways"); **cross/A shakes**, **R2 sprays** and the **left stick aims** the cursor left/right and up/down (it used to move only up and down), slower with a small push, for tracing the stencil. On the keyboard, while T or G hold the controller sideways, the **arrow keys** aim.
- **Being tested: flickering GPU smoke and particles (#7, #50).** Shader recompiler fixes, partly from the native [Shadow of the Colossus](https://github.com/AjncSystem/ShadowOfTheColossus-PC) port:
  - shared memory (LDS) barriers in compute shaders of every workgroup size, including those that are not multiples of 64 (96, 192...): `ISS_LDS_SMALL` (under 64 threads), `ISS_LDS_MULTI` (over 64; on 32-lane GPUs the barrier covers the whole workgroup, because each 64-lane PS4 wave takes two subgroups), `ISS_LDS_READLANE`;
  - `ISS_LANE_DIVERGENCE`: branches that depend on `v_mbcnt`, the lane number or an atomic's returned value count as divergent, so no barrier lands inside a branch only one lane takes;
  - `S_WQM_B64`, which did nothing (`ISS_WQM`).
  - All on by default in inFAMOUS; `0` turns each one off, for comparisons. The log has one `shader switch` line per switch.
  - Experiment, off: `ISS_RFL_ALL=1` makes `ReadFirstLane` read the first lane of the whole 64-lane wave in every wide compute shader (not only those that use ballots).
  - The shader cache rebuilds itself for each combination.
- **Fixes from [shadPS4-Infamous-FixBuild](https://github.com/rayhlancaner-png/shadPS4-Infamous-FixBuild):**
  - **Delsin's glow while draining energy fixed:** stencil now honors both the test and the write values, which are separate on PS4.
  - **More FPS:** when the game reads GPU results, only the thread that asked waits, not the whole GPU (`ISS_DEFERRED_READBACK`, on by default), and work goes to the GPU in pieces (`ISS_SUBMIT_DRAWS`, default 1024).
  - **Fewer hitches when a new effect shows up:** compute shaders compile off the GPU thread (`ISS_COMPUTE_ASYNC_BUILD`), and pipeline creation is retried when the driver fails.
  - **VRAM:** when video memory runs short, the port frees old textures and tries again instead of closing. Huge uploads, such as loading a save, go in parts, and the driver may page memory out to RAM.
  - **Float atomics** fixed on GPUs without native support, such as NVIDIA.
- **Motion sensor:** the DualShock 4/DualSense gyro works as is. On controllers without one, a tap on L3+R3 (or Share on an Xbox Series) toggles "sensor mode", the graffiti's spray can: the controller is held on its side, cross/A shakes, R2 sprays and the left stick aims. On the keyboard, T/G turn it sideways (the arrow keys aim while they are held) and Y shakes. This is what gets you past the graffiti ("Hold Wireless Controller Sideways").
- **Sticks:** a 10% dead zone against drift (`ISS_STICK_DEADZONE`, also in the launcher), and the last controller used is the one that counts.
- **Touchpad without a touchpad:** hold View/Select (or Tab) and move the right stick (or the arrow keys) to swipe, as in "Open Door"; a quick tap still clicks.
- **Performance:** FPS no longer drops over time (it went from 60 to 6 in 2 minutes: shader variants were never reused), and repeated partial image transitions are skipped.
- **Precise GPU readbacks by default:** the energy-drain smoke, drones and other GPU-computed particles show up. The fast mode is still in the launcher ("GPU accuracy").
- **Automatic shader cache:** it lives in `user/cache/`, next to the saves (Linux and Windows; the old cache in `~/.local/share/shadPS4` or `user/user/` is moved over on the first start). The folder is keyed by the shader recompiler version, and the old cache is deleted on its own after an update. To rebuild everything, the launcher has a **Clear shader cache** button (or `./run_infamous.sh --clear-shader-cache`); it deletes only this game's compiled shaders, never saves or settings.

## Status

| Area | State |
|---|---|
| Boot, title, intro and gameplay | ✅ Works; tested up to "Go to Longhouse", with climbing, running, powers and cutscenes |
| Motion sensor (graffiti, motion prompts) | ✅ Real gyro, or simulated on any controller and the keyboard |
| Drain smoke, drones and GPU particles | ✅ With precise GPU readbacks (default) |
| HUD | ✅ Correct |
| Audio | ✅ Our own AudioOut + Ajm, no firmware file needed |
| Saves | ✅ The save is created and "Continue" works |
| Controller | ✅ DualSense, DualShock 4, Xbox and 8BitDo via SDL3 |
| Performance | 🔧 20–30 FPS typical on an RTX 2060, unstable; optimization ongoing |
| Stutter | 🔧 The first time each shader is compiled (cutscenes and new areas) |
| Lights (flickering headlights, glare showing through characters) | ✅ Fixed by occlusion mode 3 (default), confirmed on Windows and on Linux; if light glare looks weak, use mode 0 |
| Graphics | 🔧 Some fog, water and fire effects still flicker |
| Launcher | ✅ Qt Quick (default) with the Tk one as a fallback; Portuguese and English |
| Saves and data | ✅ Configurable folder, backup, export/import |
| Mouse (camera and touchpad) | 🧪 Implemented, not yet tested much in game |
| Experimental fixes (BP01–BP12) | 🧪 The ones on by default come in the Recommended profile; BP07 and BP09 (off) are experimental; BP06 and BP08 are on by default |
| Linux | ✅ Tested (Nobara/Fedora 44) |
| Windows | 🧪 Tested by players (the lights fix was confirmed); known crash on RX 580/Polaris, see [Known problems](#known-problems) |

![Performance over a 7-minute session walking through the city](docs/screenshots/desempenho.png)

<sub>FPS, worst frame and draws per frame every 5 s, from the port's log (`ISS_FRAME_STATS=1`), on an RTX 2060: median 27 FPS, above 30 FPS 44% of the time. The game has no frame cap. Made with `scripts/session_chart.py` (`0` = uncapped).</sub>

## Download

The [**latest release**](https://github.com/HayatoG/iss_pc/releases/latest) is published when the maintainer chooses to bundle the accumulated changes. PRs build and test; approvals and merges do not trigger builds. On `main`, Linux and Windows build only through an authorized manual release run. Older ones stay under [Releases](https://github.com/HayatoG/iss_pc/releases) as history. See [how to publish](docs/RELEASES.md):

- **Linux x86-64** (glibc 2.39+: Ubuntu 24.04, Fedora 40, SteamOS 3.6 or newer): `infamous-linux-x86_64.tar.gz`
- **Windows 10 (1903 or newer) / 11, 64-bit**: `infamous-windows-x86_64.zip`

The packages open the Qt launcher by default; it adds about 55 MB to the download (measured on Linux; an estimate on Windows). If the new launcher does not open on your machine, use **inFAMOUS Second Son (classic)**, in the same folder.

## Required game files

Everything comes from **your own** PS4 (GoldHEN + Itemzflow). None of it is in the download. Use **inFAMOUS Second Son CUSA00223 (US), base game 1.00 with update 1.07**, merged into one folder, with the base game and the update on top. About 33 GB.

| File | From | Purpose |
|---|---|---|
| `eboot.bin` | update 1.07 | the game code (must be the 1.07 one) |
| `sce_module/libc.prx`, `sce_module/libSceFios2.prx` | update 1.07 | the game's own libraries |
| `sce_sys/` (`param.sfo`, `playgo-chunk.dat`, `icon0.png`, `pic1.png`) | base game (+ the update's `param.sfo`) | metadata, install info and the launcher art |
| `art/` (`cache/`, `movies/`, `ui/`) | base + update | game content |
| `playgo.pgm`, `sce_discmap.plt`, `sce_discmap_patch.plt` | base + update | file map and install info |

**From the console firmware:** nothing. Audio uses our own AudioOut + Ajm; the game imports neither `libSceNgs2` nor any other PS4 system module.

The full list and the step-by-step guide, which also go in the notes of every Release, are in [`docs/ARQUIVOS_NECESSARIOS.md`](docs/ARQUIVOS_NECESSARIOS.md).

**PC:** 64-bit Windows 10 (1903+)/11 or x86-64 Linux (glibc 2.39+), a graphics card with Vulkan 1.3 (6 GB of VRAM recommended) and ~33 GB free.

## How to play

1. Extract the package.
2. Open **inFAMOUS Second Son** (on Windows, if "Windows protected your PC" appears, click *More info*, then *Run anyway*: the program is not digitally signed). Pick the game folder and click **Play**. The game language follows your system by default; you can change it on the same screen. If it stutters too much, the "GPU accuracy" option (**Graphics** screen) has a fast mode (without GPU particles).
3. In game, **F10** toggles the FPS counter, and **Insert** (or R3 + L2 on a controller) opens the port's menu.
4. When the game asks you to turn or shake the controller and yours has no sensor: tap L3+R3 (or Share on an Xbox Series). The controller is then on its side; cross/A shakes, R2 sprays and the left stick aims. Tap again to leave. On the keyboard: T/G (with the arrow keys to aim) and Y. See [Motion sensor and touchpad](#motion-sensor-and-touchpad).

The full guide is in [`README-player.md`](port/packaging/README-player.md), which also ships inside the package.

## The launcher

The launcher is the program that opens when you run **inFAMOUS Second Son**. The new one (Qt Quick) is the default; the old one (Tk) sits next to it as **inFAMOUS Second Son (classic)**, in case the new one does not open. Both use the same settings and the same saves, and accept `--play`, `--script` and `--clear-shader-cache`.

| Screen | What it is for |
|---|---|
| **Play** | game folder, game language, detected GPU and version, fixes profile, the occlusion (lights) selector, the Continue/Play button |
| **Graphics** | window mode, VSync, FPS limit, FSR 1 and sharpening, GPU accuracy, VRAM budget, shader cache, overlay and presets (Performance, Balanced, Quality, Faithful to PS4) |
| **Controls** | stick dead zone, graffiti aiming speed, mouse (camera and touchpad mode), keyboard map and remapping |
| **Fixes** | BP01–BP12, diagnostic (A/B) switches and Insert menu options; Recommended profile and "Like the previous release" |
| **Saves & data** | data folder, list of saves, backups, `.zip` export/import; see [Saves and data](#saves-and-data) |
| **Controller guides** | Xbox, Switch Pro and 8BitDo images (Portuguese and English): save, copy, open the folder |
| **Support** | known problems for your GPU, links and the problem report; see [Logs and reports](#logs-and-reports) |
| **Mods / Cheats** | enables cheat XML files and the optional boot logo movie skip (story cutscenes remain unchanged) |
| **Advanced** | additional launcher and port settings |
| **Updates** | shows the installed version and checks the latest release (see below) |

**Interface language:** Portuguese (Brazil) or English (US), which the launcher picks from the system language and you can change. When you pick one, the launcher passes it to the game (`ISS_UI_LANG`) for the overlay.

**Game language:** the first option is **Automatic (system language)**, the default, followed by the languages your dump's `sce_sys/playgo-chunk.dat` lists, out of the PS4's 30 (US and UK English, Brazilian and European Portuguese, Latin American and Spain Spanish, French and Canadian French, German, Italian, Dutch, Polish, Turkish, Russian, Japanese, Korean, Chinese and more); without that file (FTP dumps) the 6 of CUSA00223: English (1), Português (Brasil) (17), Español (Latinoamérica) (20), Français (Canada) (22), Türkçe (19) and Русский (8). Automatic follows the system's language and region (es_ES is Spain Spanish, es_MX Latin American, pt_PT European Portuguese) and, when the game lacks that language, the closest variant (Spain/Latin American Spanish, European/Brazilian Portuguese...), else English. The choice applies on the next start and becomes `ISS_LANGUAGE` (an `ISS_LANGUAGE` you set yourself wins). **It only works if your game files have that language:** Turkish and Russian, for example, only show up if your dump has them. It is separate from the interface language.

**Updates:** the **Updates** screen makes a single HTTPS query for the latest release on GitHub (nothing of yours is sent), shows the notes in the interface language, and the **Download** button only opens the release page. **It does not download or install anything: a real automatic updater does not exist yet.** "Check on open" is off by default.

**Running from source:** needs Python 3 with PySide6 (>= 6.7):

```bash
python3 -m venv .venv-qt
.venv-qt/bin/pip install -r port/launcher/qt/requirements.txt
./infamous_launcher_qt.sh      # the old launcher: ./infamous_launcher.sh
```

On Windows: `py -m venv .venv-qt`, `.venv-qt\Scripts\pip install -r port\launcher\qt\requirements.txt` and `.venv-qt\Scripts\python port\launcher\infamous_launcher_qt.py`. Without PySide6 the entry point says how to install it. The package already ships everything frozen, no Python needed.

**Self-test:** `"inFAMOUS Second Son" --selftest` opens the launcher off screen, checks QML, fonts, icon and translations, visits every screen and prints `selftest ok` (it does not read your settings or saves and does not start the game). The CI runs it on every package. Details: [`docs/LAUNCHER_QT.md`](docs/LAUNCHER_QT.md) and [`docs/QT_PACKAGING.md`](docs/QT_PACKAGING.md).

## Experimental fixes

The launcher's **Fixes** screen (and the **Experimental fixes…** button of the Tk launcher) turns the 12 fixes `BP01` to `BP12`, brought from other PS4 ports and from shadPS4, on and off one by one. Each has the `ISS_BPnn` (or `TLG_BPnn`) switch and a `bpnn` line in `tlg.ini`; if you touch nothing the defaults hold, and they make up the **Recommended** profile. "In game" means it can also be changed from the Insert menu, on the spot; "restarts" applies on the next start (the ones that change shaders recompile the shaders on the first game, and the image may flicker or look green for a while).

| Id | What it does | Source | Default | Applies |
|---|---|---|---|---|
| BP01 | frees the frame of nested command buffers | shadPS4 #5258 | on | in game |
| BP02 | faster texture lookup (first page only) | shadPS4 #5240 | on | in game |
| BP03 | keeps EXEC after an empty divergent branch | shadPS4 c34ca21f | on | restarts, recompiles shaders |
| BP04 | reciprocal/root/log without clamp, like the hardware | shadPS4 #5346 | on | restarts, recompiles shaders |
| BP05 | saturate treats NaN like the GPU (becomes 0) | shadPS4 #5257 | on | restarts, recompiles shaders |
| BP06 | maximal reconvergence in compute shaders | native Shadow of the Colossus port | **on** | restarts, recompiles shaders |
| BP07 | async compile never skips small or indirect draws | bbport (bloodborne_pc) | **off** (experimental) | in game |
| BP08 | keeps the driver pipeline cache between sessions | Mrsuss, 0xCydral (bloodborne_pc) | **on** (since 2026-10-16) | restarts |
| BP09 | every 1D image becomes 2D with height 1 | CTR | **off** (experimental) | restarts, recompiles shaders |
| BP10 | barriers for vertex, index and indirect buffers | shadPS4 #5262 | on | in game |
| BP11 | `firstInstance` counted once in indirect draws | shadPS4 #5348 | on | restarts, recompiles shaders |
| BP12 | `EarlyFragmentTests` when the game asks for depth before the shader | shadPS4 #5264 | on | restarts, recompiles shaders |

- **Like the previous release:** `TLG_BACKPORTS=0` (or the launcher button) turns off every BP that is on by default, to compare with the version without them; an individual `TLG_BPnn` still wins. On the first start after updating, a one-time message explains this. If a scene gets worse, test this way first, then turn them back on one by one.
- **Back to recommended** clears all your choices. Only what differs from the default is written.
- **Diagnostic (A/B) switches:** `ISS_WQM`, `ISS_LDS_SMALL`, `ISS_LDS_MULTI`, `ISS_LDS_READLANE`, `ISS_LANE_DIVERGENCE`, `ISS_RFL_ALL`, `ISS_BLOCK1D`, `ISS_BC_TINY`, `ISS_NO_MSAA`, `ISS_BACKGROUND_WARMUP`, `ISS_PIPELINE_BINARIES` (still crashes the game on the NVIDIA 595 driver) and the occlusion selector (`ISS_OCCLUSION_MODE`); `0` turns off, `1` turns on. They exist to find which change causes a problem.
- BP08 (the on-disk driver pipeline cache) is on since 2026-10-16: pipeline creation gets fast enough that no draw is skipped (the grey one-frame flashes). The file is saved only when there are new pipelines and never grows past 1 GB (`ISS_DRIVER_CACHE_MAX_MB`; deleting it starts over). The rest of the shader stutter is on the [roadmap](docs/ROADMAP.md).
- Full table, with the `tlg.ini` keys and the Insert menu options: [`docs/EXPERIMENTAL_FIXES.md`](docs/EXPERIMENTAL_FIXES.md).

## Lights and occlusion

Headlights flickered and the glare of lights showed up **through** characters. The cause: the occlusion counter (`ZPASS_DONE`) that the game reads to know how much of each light is visible was a fake number that only grew. With mode **3** (zero counts, the default of this version) the problem **was fixed, confirmed on Windows and on Linux**.

| `ISS_OCCLUSION_MODE` | Effect |
|---|---|
| `0` | old behavior (fake counter; glare may show through) |
| `1` | test: lights count as hidden |
| `2` | test: lights count as fully visible |
| `3` | zero counts: **default** |

The **Light glare (occlusion)** selector is on the launcher's **Play** screen (and among the diagnostic switches); the variable also takes the `TLG_` prefix. **Risk:** in some scenes the glare of lights may look weaker with mode 3; if that bothers you, use mode 0. This is an approximation; real occlusion, measured through Vulkan, does not exist yet (see the [roadmap](docs/ROADMAP.md)). If you find a scene that looks worse, send the log and a screenshot of each mode.

## Mouse

The mouse works with or without a controller (its buttons add to the controller's). The game window captures the mouse while it has focus and the Insert menu is closed; **Esc releases it** (the cursor comes back) and **a click in the window captures it again** (that click is not a game button).

**Camera and buttons:**

| Mouse | In the game |
|---|---|
| Move | turns the camera (right stick) |
| Left button | R2 (use the power / shoot) |
| Right button | L2 (aim) |
| Middle button | R3 |
| Side buttons X1 / X2 and the wheel | free (see `ISS_KEY_MAP`); X1 is also a touchpad-mode key |
| Sensor on (sensor mode, T or G) | the mouse points the graffiti spray can instead of turning the camera |

**Touchpad mode:** hold **Left Ctrl** or the mouse's **X1 side button**. While the mode is on the camera does not turn: mouse motion moves a virtual finger over the touchpad (a small pad in the bottom-left corner of the screen shows where it is) and the buttons become the finger:

| Gesture (mouse in touchpad mode) | What the game sees | Where to use it |
|---|---|---|
| Just move | hovering finger (no touch): places the finger | before a gesture |
| Left, quick tap | touch + touchpad click | "press the touchpad" interactions |
| Left, hold still | touch + click held | absorbing/draining powers (smoke, neon...) |
| Left, drag more than 80 units | the click is released and the finger slides | "Open Door", hand scanner |
| Middle button (hold and move) | finger with no click | sliding without a click, if a gesture dislikes the click |
| Right | two fingers on both ends with the touchpad pressed | lifting the car off Reggie |

The finger starts in the middle of the touchpad and stays where you leave it. The keyboard shortcuts (Tab/Backspace, Tab + arrow keys) work as before.

**Environment variables** (`ISS_` and `TLG_` work the same; the `mouse_*` ones can also go in `tlg.ini`):

| Variable | Default | What it does |
|---|---|---|
| `ISS_MOUSE` | `1` | `0` turns the mouse off (the window no longer captures it) |
| `ISS_MOUSE_SENS` | `1.0` | camera sensitivity (0.05-10) |
| `ISS_MOUSE_INVERT_Y` | `0` | `1` inverts the Y axis |
| `ISS_MOUSE_SMOOTHING` | `0.5` | smoothing (0-0.95) |
| `ISS_MOUSE_ACCEL` | `0` | acceleration (0-2) |
| `ISS_MOUSE_DEADZONE` | `0.1` | anti-dead zone: small movements skip the game's dead zone (0-0.5) |
| `ISS_MOUSE_PAD` | `1` | `0` turns the mouse touchpad mode off |
| `ISS_MOUSE_PAD_KEY` | `Left Ctrl\|Mouse X1` | touchpad-mode key(s); SDL key names or `Mouse Left/Middle/Right/X1/X2`, separated by `\|` |
| `ISS_MOUSE_PAD_MODE` | `hold` | `toggle`: one press turns it on, another turns it off |
| `ISS_MOUSE_PAD_SENS` | `1.0` | finger speed (0.1-10) |
| `ISS_MOUSE_PAD_DRAG` | `80` | units (the touchpad is 1920 wide) of drag before the click becomes a swipe; `0` = never |
| `ISS_KEY_MAP` | — | remaps keys and mouse: `"r2=Mouse Left\|F,l2=Mouse Right\|R,r1=Mouse X2,l1=Wheel Down"`; several per button with `\|`, `none` removes all |
| `ISS_MOUSE_LOG` | `0` | `1` logs what the game gets from the mouse |

The Insert menu's **Controls** tab has the mouse camera settings, and the launcher's **Controls** screen has the same ones, the touchpad mode and the keyboard map. **The mouse is new and has not been tested much in game by other players yet**: if something misbehaves, `ISS_MOUSE=0` turns it off and a report helps.

**Keyboard (default map):** works with or without a controller (the keys add to the controller's buttons).

| Key | PS4 button | Key | PS4 button |
|---|---|---|---|
| W / S (A / D) | left stick | arrow keys | right stick |
| Space | cross | Left Shift | circle |
| E | square | Q | triangle |
| 1 | L1 | 3 | R1 |
| R | L2 | F | R2 |
| Z | L3 | C | R3 |
| Enter | Options | Tab / Backspace | touchpad click (left / right) |
| I / K / J / L | d-pad (up / down / left / right) | T / G / Y | simulated sensor mode (see below) |

**Remapping:** `ISS_KEY_MAP` (or the advanced field of the launcher's **Controls** screen) changes keys and mouse buttons: `"cross=Q,r2=Mouse Left|F"`. The buttons are `cross`, `circle`, `square`, `triangle`, `l1`, `r1`, `l2`, `r2`, `l3`, `r3`, `options`, `up`, `down`, `left`, `right`, `touchpad_left`, `touchpad_right`.

## Saves and data

By default saves and settings live in `user/`, next to the game. **Updating deletes nothing** (the zip does not carry `user/`), but if you extract the new zip into **another folder**, or delete the old one, the game opens without saves: they are still in the old folder, under `user/savedata`. That is why it looks like "the update deleted my save".

**Recipe to update without losing saves:**
1. Open the launcher, go to **Saves & data** and click **Back up now** (and, if you like, **export all** to a `.zip`).
2. Pick a fixed data folder outside the game (suggestion: `Documents\inFAMOUS-PC` on Windows, `~/.local/share/infamous-pc` on Linux). The launcher asks whether to copy your saves and settings there (it copies, never moves).
3. Download and extract the new version anywhere: it finds the saves through that choice, which is kept in a per-user file and survives reinstalls.
4. If you already extracted the new version and the save is gone, the screen finds the other installs (Downloads, Desktop, Documents, neighbor folders...) and offers **Import**.

**Where the data folder is**, in this order: the `ISS_USER_DIR` variable; the per-user pointer file (`%APPDATA%\inFAMOUS-PC\launcher.json` on Windows, `~/.config/infamous-pc/launcher.json` on Linux); and, with neither, `user/` next to the game (portable mode). The same folder holds `tlg.ini`, saves, backups and the shader cache.

**On the Saves & data screen:** one card per slot (title, chapter, karma, date, size and the save's own thumbnail) with `.zip` export, duplicate and delete; `.zip` import; backups with **Restore**, keeping the last N (default 10) and an extra backup folder (another disk). **A backup is made before every play session** (skipped if nothing changed). Every operation that writes to a save needs the game closed and makes a safety backup first; an invalid or dangerous `.zip` (paths with `..`, absolute paths, links) is refused without writing anything.

## Motion sensor and touchpad

The game asks for the motion sensor and the touchpad in many scenes: graffiti, absorbing powers (smoke, neon...), opening doors and other interactions. On a **DualShock 4 or DualSense** the gyro and the touchpad work as is, with nothing to turn on. On other controllers and on the keyboard, the port simulates both.

**Controller guides** (the same images are on the launcher's **Controller guides** screen, with buttons to save and copy):

| Xbox / Xbox Series | Nintendo Switch Pro | 8BitDo |
|---|---|---|
| ![Guide for Xbox, in English](port/launcher/qt/assets/controller_guides/inFAMOUS-controle-xbox-en-US.png) | ![Guide for Switch Pro, in English](port/launcher/qt/assets/controller_guides/inFAMOUS-controle-switch-en-US.png) | ![Guide for 8BitDo, in English](port/launcher/qt/assets/controller_guides/inFAMOUS-controle-8bitdo-en-US.png) |

**Sensor mode (turning it on and off):**
- **One tap on L3+R3 together** turns sensor mode on; another tap turns it off. It works on any controller. The controller's extra button works too: **Share** on an Xbox Series, **Capture** on a Switch Pro and the **microphone** button on a DualSense. On a DualShock 4 or DualSense, Share/Create does **not** toggle sensor mode: it is the touchpad click.
- A notice on screen confirms it: "Sensor mode ON (spray can): cross shakes, R2 sprays, left stick aims" or "Sensor mode off".
- While it is on, the controller is **on its side**, like a spray can (what the graffiti asks for), **cross/A shakes** and the **left stick aims**, pointing the can left/right and up/down (this moves the graffiti cursor; it stays where you let go; a small push is slow, for tracing the stencil; the right stick works too). R2 still sprays. The sticks and cross/A are hidden from the game, so Delsin neither walks nor jumps and the camera does not turn: **turn the mode off after the scene**. L3+R3 together never reaches the game either.
- While it is on, the simulated tilt replaces the real gyro.
- **Keyboard** (no mode needed, any time): hold **T** or **G** to turn the controller sideways (one side or the other) and **Y** to shake it.

**Touchpad on a controller without one:**
- **View/Select/Back** is the touchpad click (right side; with L1 held, the left side). Holding the button holds the touchpad.
- **Hold View/Select and move the right stick** to swipe a finger from the middle of the touchpad. A quick tap without moving still clicks.
- **Keyboard:** Tab (left side) and Backspace (right side) click; **hold Tab and use the arrow keys** to swipe.
- **Two fingers on both ends of the touchpad** (lifting the car off Reggie): **hold View/Select + R1**; on the keyboard, **hold Tab + Backspace**.

| In-game action | DualShock 4 / DualSense | Xbox / other controller | Keyboard |
|---|---|---|---|
| Absorb/drain a power (smoke, neon...) and "hold the touchpad" interactions | hold the touchpad | hold View/Select (leave the right stick alone, or it becomes a swipe) | hold Tab or Backspace (no arrow keys); mouse: Left Ctrl (or X1) + hold the left button still |
| "Press the touchpad" interactions | click the touchpad | tap View/Select | Tab or Backspace |
| Graffiti: turn the controller sideways | turn the controller | L3+R3 (sensor mode): it is on its side at once | hold T or G |
| Graffiti: shake the can | shake the controller | cross/A in sensor mode | Y |
| Graffiti: spray | R2, aiming with the controller | R2, aiming with the left stick in sensor mode | F (R2), aiming with the arrow keys while holding T or G |
| Press both ends of the touchpad (lifting the car off Reggie) | two fingers, one on each end | hold View/Select + R1 | hold Tab + Backspace; mouse: Left Ctrl (or X1) + right button |
| Touchpad swipe ("Open Door", hand scanner) | swipe a finger | hold View/Select + right stick in the asked direction | hold Tab + arrow keys; mouse: hold Left Ctrl (or X1), left button and drag |
| Back to normal after the graffiti | — | L3+R3 again ("Sensor mode off") | — |

**Launcher:** the **Controller** box has the **Stick dead zone** (0–40%, default 10%). Raise it if the character, the camera or the sensor-mode tilt moves on its own (stick drift).

**Environment variables:** `ISS_MOTION_BUTTON=<SDL button>` changes the sensor-mode button (default `misc1`; L3+R3 always works), and `ISS_MOTION=off` ignores the real gyro (the simulation still works).

**Intel CPUs:** the game's code uses SSE4A instructions (EXTRQ), which only AMD CPUs have. On Intel the port replaces them with an emulation at start (needs `iced-x86`, included in the packages). Older CPUs (Sandy and Ivy Bridge) also lack MOVBE, F16C, BMI1 or LZCNT, which are emulated the same way; AVX is still required. `ISS_ISA_REWRITE=sse4a` (or `f16c`, or a list such as `f16c,movbe`) forces the emulation on a CPU that has the instruction (testing), `ISS_ISA_REWRITE=1` forces every emulated set (MOVBE, BMI1, LZCNT, F16C, SSE4A) and `ISS_ISA_REWRITE=0` turns it off. Details: [port/docs/isa_rewrite.md](port/docs/isa_rewrite.md).

**Windows red-zone protection.** PS4 code keeps variables in the 128 bytes below `rsp` (the "red zone"), and Windows writes there on every exception (the port uses exceptions to track GPU memory). At start the port rewrites the memory instructions of the functions that use the red zone so they run with `rsp` 128 bytes lower (crash at `0x1cfaaf`, #29/#41). On by default on Windows; `ISS_RED_ZONE_REWRITE=0` turns it off and `ISS_RED_ZONE_REWRITE=1` turns it on on Linux (testing). Linux test: `TLG_CLOBBER_RED_ZONE=<bytes>` zeroes that many bytes below `rsp` on every exception, like Windows (`1` = 128). Details: [port/docs/red_zone.md](port/docs/red_zone.md).

**Frozen screen:** if the game shows no new frame for 10 s, the log (`out/last_run.log`) gets a `FREEZE: no frame for 10 s; threads:` block with every thread of the game, what it is waiting on (mutex, semaphore, event queue, file read, GPU...) and where it is in the code. The game keeps running, and the report only repeats after frames resume and stop again. Send that log with the report. `ISS_FREEZE_REPORT_SECONDS=<s>` changes the time (`0` turns it off).

**AMD cards on Windows (RX 580): experimental compatibility.** Players have reported crashes after the logos in `amdvlk64.dll+0xebbbb` (#31, #49). The [PR #232](https://github.com/HayatoG/iss_pc/pull/232) investigation found a batch of 9,217 barriers, including 9,216 for one image with 1,536 layers and 6 mips; the log does not prove that call caused the crash. Two players reported getting past that point with a test build. Enable **RX580 compatibility** in **Graphics → Image**, or choose **RX 580 compatibility (experimental)** to also apply the recommended AMD settings. This test defaults to off, requires a game restart and still needs validation in this version. The same switch appears in **Advanced → Image workarounds**. From a terminal: `ISS_BARRIER_MERGE=1` merges equivalent adjacent image barriers; `0` disables it. Values set outside the launcher take precedence. The earlier tiny-BC texture workaround remains available (`ISS_BC_TINY=0/1`), but is not a confirmed fix for this crash. For diagnostic comparisons, **Advanced** exposes `ISS_EXPLICIT_RANGE` (default `1`, explicit mip/layer counts) and `ISS_BARRIER_SPLIT` (default `0`; `1` splits per mip, `2` per mip and layer; requires explicit ranges). These experiments alone did not resolve the crash in the author's tests. Credit: @intemerated for merging and investigation; @Andretuta for the initial launcher integration.

## Logs and reports

Every game writes a log to `out/logs/run-YYYYMMDD-HHMMSS.log` (the most recent ones are kept; `out/last_run.log` points to the latest). The log starts with the `Build:` line and a **`Config:`** block with the keys of your `tlg.ini` and the `TLG_`/`ISS_` variables that are set (values that look like paths are left out), and has the `Backports:` lines (each BP with its on/off state, and the occlusion mode) and, with the frame statistics, `Backport hits (total): ...` (how many times each fix actually ran). With that, a report says which fixes were on.

**How to report a problem:** open the launcher's **Support** screen and use **Report a problem**: it collects GPU, driver, VRAM, system, CPU, port and game versions, lets you pick the log, **strips personal paths from the text** (user name and folder) and offers **Copy report**, **Save .zip** (report and log without paths) and **Open issue on GitHub**. Nothing is sent without your click. To report by hand, attach the log to an [issue](https://github.com/HayatoG/iss_pc/issues) or send it on [Discord](https://discord.gg/UMkdMYZh9g).

**Big log:** GitHub refuses big files. Compress the log into a `.zip` (the Support screen already does) before attaching it.

## Known problems

The launcher's **Support** screen shows the ones that match your GPU (a local table, no internet query). The ones known today:

| Situation | What happens | What to do |
|---|---|---|
| **AMD RX 580 / Polaris on Windows** (#31, #49, #59, #65, #67, #81) | crash right after the logos in `amdvlk64.dll+0xebbbb` | **experimental workaround available.** Try the **RX 580 compatibility (experimental)** launcher preset, restart the game and send the log; a fix is not guaranteed |
| **GPU with 4 GB of VRAM** (#72, #75) | texture memory runs out; very low FPS | lower the resolution (Graphics screen) and keep the VRAM budget below your card's memory. Note: **"GPU accuracy: Fast" darkens the image** |
| **NVIDIA Quadro / Maxwell** (#70) | may be below the port's minimum; black screen or the game does not open | tell us what you saw; no fix yet |
| **CPU without SSE4A or F16C** (Intel, Sandy Bridge...) | the port emulates those instructions at start | automatic; see [Intel CPUs](#motion-sensor-and-touchpad) |
| **Game version** | tested with **CUSA00223 1.00 + 1.07**; CUSA00004 also boots | use CUSA00223 with the 1.07 update |

NVIDIA RTX 30 cards on Windows have reported flickering lights and smoke; use the Recommended profile and play two sessions in a row (the first one recompiles the shaders). For the other open problems, see the [issues](https://github.com/HayatoG/iss_pc/issues).

## Building from source

```bash
git clone --recursive https://github.com/HayatoG/iss_pc.git && cd iss_pc
cd port && ISS_LTO=OFF bash build.sh   # Linux: SDL3, Vulkan, FFmpeg, FreeType, Boost, fmt, Zydis, glslang
cd .. && ./infamous_launcher_qt.sh     # Qt launcher (needs PySide6); ./infamous_launcher.sh is the Tk one; ./run_infamous.sh from a terminal
```

A clone builds without profile-guided optimization (PGO): the profile (`port/pgo/*.gcda`) is not in the tree, and the release packages are built without it as well, so a local build and a package should perform alike. `ISS_PGO=generate` builds an instrumented binary that writes the profile while you play, and the next build uses it.

On Windows, the build runs in MSYS2 CLANG64 ([`port/packaging/windows/README.md`](port/packaging/windows/README.md)). The workflow [`.github/workflows/build.yml`](.github/workflows/build.yml) shows the full process on both platforms.

## How it works

| Layer | What it does | Based on |
|---|---|---|
| Loader | Loads the 1.07 `eboot.bin` and links the game's own modules (`libc`, `Fios2`) | [tlg_pc](https://github.com/HayatoG/tlg_pc), [gr2_pc](https://github.com/HayatoG/gr2_pc), [gow3_pc](https://github.com/FelipePrado0/gow3_pc), [bloodborne_pc](https://github.com/deadinside28/bloodborne_pc) |
| Runtime | Reimplements the PS4 libraries (NpManager, Trophy, Http, PlayGo, SaveData, AppContent, AudioOut, Ajm...). inFAMOUS-specific: `port/src/runtime_infamous.c` | `port/src/` |
| Video | Translates Gnm commands and GCN shaders to Vulkan/SPIR-V | [shadPS4](https://github.com/shadps4-emu/shadPS4) video core |

## Next steps

What is left is in [`docs/ROADMAP.md`](docs/ROADMAP.md): performance (waits on GPU readbacks), an on-disk pipeline cache against the stutter (BP08 is only the first step), real occlusion through Vulkan, the fog, water and fire effects that still flicker, the drone-drain glow, the RX 580 crash on Windows, an automatic updater and validating the mouse.

## Credits and legal notice

Code base: [tlg_pc](https://github.com/HayatoG/tlg_pc), [gr2_pc](https://github.com/HayatoG/gr2_pc), [gow3_pc](https://github.com/FelipePrado0/gow3_pc) (FelipePrado0), [bloodborne_pc](https://github.com/deadinside28/bloodborne_pc) and [shadPS4](https://github.com/shadps4-emu/shadPS4) the [shadPS4-Infamous-FixBuild](https://github.com/rayhlancaner-png/shadPS4-Infamous-FixBuild) fork (Eye Of The Ruins/rayhlancaner-png, TOXIC-KID, 0xDeftones) and the native [ShadowOfTheColossus-PC](https://github.com/AjncSystem/ShadowOfTheColossus-PC) port (AjncSystem; with parts of Pink-shadPS4 by luizgustavs), all GPL-2.0, as is this project ([LICENSE](port/LICENSE)). ATRAC9 audio via [LibAtrac9](https://github.com/Thealexbarney/LibAtrac9); fonts via [FreeType](https://freetype.org).

**Origin of the experimental fixes (BP01–BP12):** the recompiler and video fixes come from [shadPS4](https://github.com/shadps4-emu/shadPS4) (BP01–BP05, BP10–BP12), the native [Shadow of the Colossus](https://github.com/AjncSystem/ShadowOfTheColossus-PC) port (BP06), bbport, i.e. [bloodborne_pc](https://github.com/deadinside28/bloodborne_pc) (BP07), Mrsuss's [bloodborne_pc_windows_port](https://github.com/Mrsuss60/bloodborne_pc_windows_port) and 0xCydral's [bloodborne_pc](https://github.com/0xCydral/bloodborne_pc) (BP08) and CTR ([T3CHNOLOG1C/CTR](https://github.com/T3CHNOLOG1C/CTR)) (BP09). The exact commits are in [`docs/EXPERIMENTAL_FIXES.md`](docs/EXPERIMENTAL_FIXES.md) and `port/gpu/VENDOR.txt`. Launcher fonts: Big Shoulders Stencil Display and Barlow (OFL license, in `port/launcher/qt/fonts/`).

### Acknowledgements

- **Kazu**, for the logs, screenshots and tests, on Windows and on Linux, that led to the fix for the flickering headlights and the glare showing through characters (occlusion mode 3).
- The players who reported problems in the [issues](https://github.com/HayatoG/iss_pc/issues), with logs that changed the port's course: jacckkss (#50, flickering lights and smoke), heyzerr (#51, #65, #67, CPUs without F16C and the RX 580), Andrade2244 (#29), MrRobot11214 (#31), fghacksr (#49), AngelHomura (#59) and aledieb93-svg (#81) (crashes on the RX 580 and on Windows), ViniSnapeS (#72) and kudetgaming-art (#75) (video memory), summer882 (#71), manymanyyyd-afk (#63) and Kpopping (#73) (Turkish and Russian), mahmoudmahran2112-crypto (#70, Quadro/Maxwell), Milicksp (#16), gsev3333 (#41) and everyone else who opened an issue or tested a build.

This project is not affiliated with Sony Interactive Entertainment or Sucker Punch Productions. inFAMOUS is a trademark of Sony Interactive Entertainment. No game code, files or art are distributed here. The icon is original artwork made for this project.

## Apoie o desenvolvimento / Support development

Se quiser apoiar os ports, [pague um café pelo Buy Me a Coffee](https://buymeacoffee.com/hayatog). O apoio é opcional; os ports continuam gratuitos.

If you would like to support the ports, [buy me a coffee](https://buymeacoffee.com/hayatog). Support is optional; the ports remain free.
