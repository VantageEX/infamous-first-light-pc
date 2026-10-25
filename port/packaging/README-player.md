# inFAMOUS Second Son — port para PC / PC port

Por HayatoG · YouTube: [@hayatog-dev](https://www.youtube.com/@hayatog-dev) · Discord: [discord.gg/UMkdMYZh9g](https://discord.gg/UMkdMYZh9g) · Código: [github.com/HayatoG/iss_pc](https://github.com/HayatoG/iss_pc)

**Nenhum arquivo do jogo vem neste pacote.** Você precisa da sua própria cópia de inFAMOUS Second Son para PS4. / **No game files are included.** You need your own copy of inFAMOUS Second Son for PS4.

Jogável, em desenvolvimento: o jogo inicia e chega ao gameplay, mas o desempenho ainda é instável e há falhas de imagem conhecidas. / Playable, in development: the game boots and reaches gameplay, but performance is still unstable and there are known graphics glitches.

---

## Português

### Arquivos do jogo necessários

Tudo vem do **seu** PS4 (GoldHEN + Itemzflow). Nada disso vai no download. Use **inFAMOUS Second Son CUSA00223 (EUA), jogo base 1.00 com o update 1.07**, montado numa pasta só, com o jogo base e o update por cima. São ~33 GB.

| Arquivo | Vem de | Para quê |
|---|---|---|
| `eboot.bin` | update 1.07 | o código do jogo (tem que ser o do 1.07) |
| `sce_module/libc.prx`, `sce_module/libSceFios2.prx` | update 1.07 | bibliotecas do próprio jogo |
| `sce_sys/` (`param.sfo`, `playgo-chunk.dat`, `icon0.png`, `pic1.png`) | jogo base (+ `param.sfo` do update) | metadados, instalação e a arte do launcher |
| `art/` (`cache/`, `movies/`, `ui/`) | base + update | conteúdo do jogo |
| `playgo.pgm`, `sce_discmap.plt`, `sce_discmap_patch.plt` | base + update | mapa de arquivos e instalação |

**Do firmware do console:** nada. A pasta `system_lib/` fica vazia.

**Como montar a pasta:**
1. No Itemzflow, faça o dump do jogo (`CUSA00223`) e do update (`CUSA00223-patch`) para um pendrive exFAT.
2. No PC, copie a pasta do jogo base e depois copie o conteúdo do update **por cima**, substituindo os arquivos.
3. O `param.sfo` da versão 1.07 fica no PS4 em `/system_data/priv/appmeta/CUSA00223/param.sfo`: copie para o `sce_sys/` da pasta montada.

**PC:** Windows 10 (1903+)/11 64 bits ou Linux x86-64 (glibc 2.39+), placa de vídeo com Vulkan 1.3 (6 GB de VRAM recomendados) e ~33 GB livres.

### Como jogar

1. Extraia o pacote numa pasta qualquer (no Windows, fora do .zip).
2. Abra **inFAMOUS Second Son** (`inFAMOUS Second Son.exe` no Windows). No Windows, se aparecer "O Windows protegeu o computador", clique em *Mais informações* e depois em *Executar assim mesmo*. O programa não é assinado digitalmente.
3. Em **Pasta do jogo**, escolha a pasta que tem o `eboot.bin` e clique em **Jogar**.

O launcher tem 10 telas, incluindo **Mods / Cheats** e **Avançado**. O launcher antigo continua na mesma pasta como **inFAMOUS Second Son (classic)**: use-o se o novo não abrir.
- **Pular vídeo inicial:** na tela **Mods / Cheats**, ative **Skip intro movie**. Isso pula apenas o vídeo de logos antes do menu; não pula as cutscenes da história. Vem desligado por padrão.

- **Idioma do jogo:** por padrão é o do seu sistema ("Automático"); dá para escolher English, Português (Brasil), Español (Latinoamérica), Français (Canada), Türkçe ou Русский na tela **Jogar**. Só funciona se os arquivos do seu jogo tiverem o idioma. A interface do launcher é em português ou inglês.
- **Saves:** ficam em `user/`, ao lado do jogo. A atualização não apaga nada, mas **extrair o zip novo em outra pasta faz o jogo abrir sem saves** (eles continuam na pasta antiga, em `user/savedata`). Para atualizar sem perder nada, abra **Saves e dados**, faça um backup e escolha uma pasta fixa fora do jogo (a tela copia seus saves para lá); ela também acha saves de outras pastas, importa e exporta `.zip` e faz backup antes de cada partida.
- **Luzes:** os faróis que piscavam e o brilho que atravessava personagens foram corrigidos (seletor "Brilho das luzes (oclusão)" na tela **Jogar**; padrão 3). Se o brilho das luzes ficar fraco, escolha 0.
- **Correções experimentais:** a tela **Correções** liga e desliga 12 correções (BP01–BP12) e chaves de diagnóstico. O padrão é o perfil Recomendado; "Como a release anterior" desliga as novas para comparar.
- **Atualizações:** a tela só verifica se há uma release mais nova e abre a página para baixar; não instala nada.

No Linux, `./install_desktop.sh` adiciona o jogo ao menu de aplicativos.

### Durante o jogo
- **F10**: contador de FPS · **F9**: teclas e botões do mouse pressionados · **Insert** (ou R3 + L2): menu do port
- Controles de PS4/PS5, Xbox e 8BitDo via SDL3, por USB ou Bluetooth.

### Sensor de movimento e touchpad
Para o grafite, absorver poderes (fumaça, neon...), abrir portas e outras cenas de sensor ou touchpad. No DualShock 4/DualSense, o giroscópio e o touchpad funcionam direto.
- **Modo sensor:** um toque em **L3+R3** juntos liga, outro toque desliga (ou Share no Xbox Series, Capture no Switch Pro, microfone no DualSense). Ligado, ele é a lata de spray do grafite: o controle fica **de lado**, **X/A sacode**, **R2 pinta** e o **analógico esquerdo mira** (o cursor do grafite); o Delsin não anda nem pula, então desligue depois da cena. Um aviso na tela mostra o estado.
- **Touchpad:** **View/Select** clica (segurar = segurar o touchpad, para absorver poderes); **segure View/Select + analógico direito** para deslizar (porta "Open Door", scanner de mão); **View/Select + R1** aperta as duas pontas (levantar o carro de cima do Reggie).
- **Mouse:** mover gira a câmera; esquerdo = R2, direito = L2, do meio = R3; **Esc** solta o mouse e um clique prende de novo. **Segure Ctrl esquerdo (ou o botão lateral X1)** para usar o mouse como touchpad: esquerdo toca/clica (segurar = absorver poder), esquerdo + arrastar desliza ("Open Door", scanner), direito = dois dedos (carro do Reggie). `ISS_MOUSE=0` desliga.
- **Teclado:** **T/G** viram o controle de lado (enquanto seguram, as **setas** miram), **Y** sacode, **F** é o R2 (pintar); **Tab/Backspace** clicam o touchpad, **Tab + setas** deslizam, **Tab + Backspace** apertam as duas pontas.
- **Launcher:** "Zona morta dos analógicos" (padrão 10%), se algo se mexer sozinho. A tela **Guias de controle** tem imagens Xbox, Switch Pro e 8BitDo desse passo a passo, para salvar ou copiar.
- O mouse é novo e foi pouco testado em jogo: se der problema, `ISS_MOUSE=0` o desliga.

### Problemas
- **Não abre ou fecha sozinho**: tela **Suporte** > **Relatar um problema** monta o relatório (GPU, driver, versão) com o log **sem caminhos pessoais**, e deixa copiar, salvar um `.zip` ou abrir uma issue. Os logs ficam em `out/logs/run-*.log` e começam com um bloco `Config:` com as suas opções. Se o log for grande, compacte em `.zip` antes de anexar (o GitHub recusa arquivo grande). Mande no [Discord](https://discord.gg/UMkdMYZh9g) ou numa [issue](https://github.com/HayatoG/iss_pc/issues).
- **Placa AMD RX 580 / Polaris no Windows** fecha logo depois dos logos (`amdvlk64.dll+0xebbbb`): a causa ainda está em investigação; atualize o driver e teste `ISS_BP09=1`.
- **GPU de 4 GB de VRAM**: a memória de textura pode estourar; baixe a resolução. A opção "Precisão da GPU: Rápido" escurece a imagem.
- **NVIDIA Quadro / Maxwell**: pode estar abaixo do mínimo do port.
- **Tela congelada (o jogo para, a janela continua aberta)**: depois de 10 s sem quadro novo, o `out/last_run.log` recebe um bloco `FREEZE:` com o que cada thread está esperando. Feche o jogo e mande esse log. `ISS_FREEZE_REPORT_SECONDS` muda o tempo (`0` desliga).

---

## English

### Required game files

Everything comes from **your own** PS4 (GoldHEN + Itemzflow). None of it is in the download. Use **inFAMOUS Second Son CUSA00223 (US), base game 1.00 with update 1.07**, merged into one folder, with the base game and the update on top. About 33 GB.

| File | From | Purpose |
|---|---|---|
| `eboot.bin` | update 1.07 | the game code (must be the 1.07 one) |
| `sce_module/libc.prx`, `sce_module/libSceFios2.prx` | update 1.07 | the game's own libraries |
| `sce_sys/` (`param.sfo`, `playgo-chunk.dat`, `icon0.png`, `pic1.png`) | base game (+ the update's `param.sfo`) | metadata, install info and the launcher art |
| `art/` (`cache/`, `movies/`, `ui/`) | base + update | game content |
| `playgo.pgm`, `sce_discmap.plt`, `sce_discmap_patch.plt` | base + update | file map and install info |

**From the console firmware:** nothing. The `system_lib/` folder stays empty.

**How to build the folder:**
1. In Itemzflow, dump the game (`CUSA00223`) and the update (`CUSA00223-patch`) to an exFAT USB drive.
2. On the PC, copy the base game folder, then copy the update's contents **over it**, replacing files.
3. The 1.07 `param.sfo` is at `/system_data/priv/appmeta/CUSA00223/param.sfo` on the PS4: copy it into the merged folder's `sce_sys/`.

**PC:** 64-bit Windows 10 (1903+)/11 or x86-64 Linux (glibc 2.39+), a graphics card with Vulkan 1.3 (6 GB of VRAM recommended) and ~33 GB free.

### How to play

1. Extract the package anywhere (on Windows, outside the .zip).
2. Open **inFAMOUS Second Son** (`inFAMOUS Second Son.exe` on Windows). On Windows, if "Windows protected your PC" appears, click *More info*, then *Run anyway*. The program is not digitally signed.
3. Under **Game folder**, choose the folder that has `eboot.bin` and click **Play**.

The launcher has 10 screens, including **Mods / Cheats** and **Advanced**. The old launcher stays in the same folder as **inFAMOUS Second Son (classic)**: use it if the new one does not open.
- **Skip the boot intro:** on **Mods / Cheats**, enable **Skip intro movie**. This skips only the logo video before the title screen; story cutscenes are unchanged. It is off by default.

- **Game language:** it follows your system by default ("Automatic"); you can pick English, Português (Brasil), Español (Latinoamérica), Français (Canada), Türkçe or Русский on the **Play** screen. It only works if your game files have that language. The launcher interface is in Portuguese or English.
- **Saves:** they live in `user/`, next to the game. Updating deletes nothing, but **extracting the new zip into another folder makes the game open without saves** (they stay in the old folder, under `user/savedata`). To update without losing anything, open **Saves & data**, make a backup and pick a fixed folder outside the game (the screen copies your saves there); it also finds saves in other folders, imports and exports `.zip`, and backs up before every play session.
- **Lights:** the flickering headlights and the glare showing through characters are fixed ("Light glare (occlusion)" selector on the **Play** screen; default 3). If light glare looks weak, pick 0.
- **Experimental fixes:** the **Fixes** screen turns 12 fixes (BP01–BP12) and diagnostic switches on and off. The default is the Recommended profile; "Like the previous release" turns the new ones off to compare.
- **Updates:** the screen only checks whether a newer release exists and opens the page to download it; it installs nothing.

On Linux, `./install_desktop.sh` adds the game to the applications menu.

### While playing
- **F10**: FPS counter · **F9**: keys and mouse buttons held · **Insert** (or R3 + L2): the port's menu
- PS4/PS5, Xbox and 8BitDo controllers via SDL3, over USB or Bluetooth.

### Motion sensor and touchpad
For the graffiti, absorbing powers (smoke, neon...), opening doors and other sensor or touchpad scenes. On a DualShock 4/DualSense, the gyro and the touchpad work as is.
- **Sensor mode:** one tap on **L3+R3** together turns it on, another turns it off (or Share on an Xbox Series, Capture on a Switch Pro, the microphone button on a DualSense). While on, it is the graffiti's spray can: the controller is **on its side**, **cross/A shakes**, **R2 sprays** and the **left stick aims** (the graffiti cursor); Delsin neither walks nor jumps, so turn it off after the scene. A notice on screen shows the state.
- **Touchpad:** **View/Select** clicks (hold = hold the touchpad, to absorb powers); **hold View/Select + right stick** to swipe ("Open Door", hand scanner); **View/Select + R1** presses both ends (lifting the car off Reggie).
- **Mouse:** moving turns the camera; left = R2, right = L2, middle = R3; **Esc** releases the mouse and a click captures it again. **Hold Left Ctrl (or the X1 side button)** to use the mouse as the touchpad: left touches/clicks (hold = absorb a power), left + drag swipes ("Open Door", scanner), right = two fingers (Reggie's car). `ISS_MOUSE=0` turns it off.
- **Keyboard:** **T/G** turn the controller sideways (while held, the **arrow keys** aim), **Y** shakes, **F** is R2 (spray); **Tab/Backspace** click the touchpad, **Tab + arrow keys** swipe, **Tab + Backspace** press both ends.
- **Launcher:** "Stick dead zone" (default 10%), if something moves on its own. The **Controller guides** screen has Xbox, Switch Pro and 8BitDo images of these steps, to save or copy.
- The mouse is new and has not been tested much in game: if it misbehaves, `ISS_MOUSE=0` turns it off.

### Problems
- **Does not start or closes by itself**: the **Support** screen > **Report a problem** builds the report (GPU, driver, version) with the log **without personal paths**, and lets you copy it, save a `.zip` or open an issue. Logs are in `out/logs/run-*.log` and start with a `Config:` block with your options. If the log is big, compress it into a `.zip` before attaching (GitHub refuses big files). Send it on [Discord](https://discord.gg/UMkdMYZh9g) or in an [issue](https://github.com/HayatoG/iss_pc/issues).
- **AMD RX 580 / Polaris on Windows** closes right after the logos (`amdvlk64.dll+0xebbbb`): the cause is still under investigation; update the driver and try `ISS_BP09=1`.
- **GPU with 4 GB of VRAM**: texture memory can run out; lower the resolution. The "GPU accuracy: Fast" option darkens the image.
- **NVIDIA Quadro / Maxwell**: may be below the port's minimum.
- **Frozen screen (the game stops, the window stays open)**: after 10 s without a new frame, `out/last_run.log` gets a `FREEZE:` block with what each thread is waiting on. Close the game and send that log. `ISS_FREEZE_REPORT_SECONDS` changes the time (`0` turns it off).

---

This project is not affiliated with Sony Interactive Entertainment or Sucker Punch Productions. inFAMOUS is a trademark of Sony Interactive Entertainment. / Este projeto não tem vínculo com a Sony Interactive Entertainment ou a Sucker Punch Productions.

## Apoie o desenvolvimento / Support development

☕ [Apoie com um café / Buy me a coffee](https://buymeacoffee.com/hayatog)

Apoio opcional para continuar desenvolvendo os ports. Os ports continuam gratuitos.
Optional support for continued port development. The ports remain free.
