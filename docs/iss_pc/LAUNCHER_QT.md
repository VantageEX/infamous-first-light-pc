# Launcher Qt Quick (o padrão dos pacotes)

O launcher usa Qt Quick (PySide6) e é o **padrão** dos pacotes (veja [QT_PACKAGING.md](QT_PACKAGING.md)); o launcher Tk (`infamous_launcher.py`) continua no pacote como **inFAMOUS Second Son (classic)**, de reserva. O trilho de navegação à esquerda tem oito telas: **Jogar** (com o botão Jogar, o idioma do jogo e o seletor de oclusão), **Gráficos**, **Controles**, **Correções** (BP01-BP12 e chaves A/B), **Saves e dados**, **Guias de controle**, **Suporte** e **Atualizações**. A interface vem em pt-BR e en-US (idioma do sistema por padrão).

Ele **não reimplementa nada**: usa `infamous_run.py` (`prepare`, `launch`, `build_env`, `save_experimental`), `backports_table.py`, `shader_cache.py` e `tlg_vulkan.py`. O mesmo `settings.json` e o mesmo `tlg.ini` valem nos dois launchers.

## Como rodar

```bash
python3 -m venv .venv-qt
.venv-qt/bin/pip install -r port/launcher/qt/requirements.txt   # PySide6-Essentials >= 6.7
./infamous_launcher_qt.sh        # ou: python3 port/launcher/infamous_launcher_qt.py
```

No Windows: `py -m venv .venv-qt`, `.venv-qt\Scripts\pip install -r port\launcher\qt\requirements.txt` e `.venv-qt\Scripts\python port\launcher\infamous_launcher_qt.py`. Os argumentos `--play`, `--script` e `--clear-shader-cache` funcionam como no launcher Tk.

Sem PySide6 o ponto de entrada imprime como instalar e sai. As fontes (Big Shoulders Stencil Display e Barlow, licença OFL, arquivos `OFL-*.txt`) ficam em `port/launcher/qt/fonts/` e são carregadas na execução; sem elas, cai a fonte do sistema.

## Idioma do jogo (Início)

Seletor "Idioma do jogo / Game language" no canto do herói, sob **Escolher a pasta do jogo**. A primeira opção é **Automático (idioma do sistema)**, o novo padrão (`settings['game_language'] = 'auto'`); depois vêm as línguas de `infamous_run.GAME_LANGUAGES` (os 30 idiomas do PS4, cada um com o seu id) que o dump tem, segundo os chunks PlayGo de `sce_sys/playgo-chunk.dat` (`game_language_choices`); sem esse arquivo (dumps por FTP) aparecem as 6 do CUSA00223 (`_DEFAULT_LANGUAGES`). Com "auto" ativo aparece embaixo "Automático: <idioma que vale>".

`infamous_run.system_game_language()` lê `LC_ALL`, `LC_MESSAGES`, `LANG`, depois `locale.getlocale()` e, no Windows, `GetUserDefaultLocaleName`; pelo prefixo: pt (inclui pt_PT) vira `pt_BR`, es `es_LA`, fr `fr_CA`, tr `tr`, ru `ru`, o resto `en`. `resolve_game_language()` devolve a escolha explícita, ou o idioma do sistema para `auto` e valores inválidos. `build_env` usa essa função (`ISS_LANGUAGE` do ambiente continua ganhando): en=1, pt_BR=17, es_LA=20, fr_CA=22, tr=19, ru=8. Quem já tinha `pt_BR` (ou outro) gravado mantém. O launcher Tk usa a mesma caixa e a mesma função.

Escolher grava a chave por `save_settings` e vale na próxima partida; a ajuda diz isso e que só funciona se os arquivos do jogo tiverem o idioma. É separado da língua da interface (PT-BR / EN-US). Código: `Backend.gameLanguageOptions/gameLanguageIndex/gameLanguageNow/setGameLanguage`.

## Oclusão (Início)

Seletor **Brilho das luzes (oclusão)** na tela Início, com os quatro valores de `ISS_OCCLUSION_MODE` (0 como antes, 1 luzes tapadas, 2 luzes totalmente visíveis, 3 contagem zero, o padrão); grava `occlusion_mode` pela mesma tabela da tela Correções (`Backend.occlusionOptions/occlusionIndex/setOcclusion`) e vale na próxima partida. O modo 3 corrigiu os faróis que piscavam e o brilho que atravessava personagens (confirmado no Windows e no Linux); veja o [README](../README.md#luzes-e-oclusão). O launcher Tk tem a mesma caixa.

## O que a tela Início lê

| Item | De onde |
|---|---|
| GPU e Vulkan | `vulkaninfo --summary`, com o mesmo ambiente do jogo (`tlg_vulkan.configure`), numa thread |
| Pasta do jogo e versão | `infamous_run.check_game` (aceita só o update 1.07) |
| Perfil | `backports_table.changes`: Recomendado ou personalizado (N mudanças) |
| Cache de shaders | `infamous_run.shader_cache_size` |
| Controle | `/proc/bus/input/devices` (Linux; no Windows o chip diz que não há detecção) |
| Continuar | `param.bin`/`param.sfo` mais recente em `user/savedata` (SUBTITLE e DETAIL); sem save, o botão só diz "Iniciar o jogo" |
| Novidades | as 3 primeiras linhas de "Novidades" do `README.md` (ou texto fixo) |
| Arte do herói | `sce_sys/pic1.png` do dump do jogador, lido em tempo de execução; sem ele, o gradiente de neon e chuva desenhado em `qt/app.py`. Nada da Sony vai para o repositório |

## Saves e dados

Entrada **Saves e dados** no trilho (destaque verde). O problema que resolve: com os dados em `<pasta do port>/user`, quem atualiza extraindo o zip em outra pasta (ou dentro da antiga, ou apagando a antiga) abre o jogo sem saves e acha que a atualização os apagou. A atualização nunca apaga nada (o zip não traz `user/`), mas os saves ficaram na pasta antiga.

**Pasta de dados** (`infamous_run.user_dir()`, regras em `launcher/userdata.py`), nesta ordem:

1. variável de ambiente `ISS_USER_DIR` (`TLG_USER_DIR` ainda vale);
2. arquivo-ponteiro por usuário, **fora** da pasta do jogo: Windows `%APPDATA%\inFAMOUS-PC\launcher.json`, Linux `$XDG_CONFIG_HOME/infamous-pc/launcher.json` (chaves `user_dir`, `backup_extra_dir`, `backup_keep`);
3. `<pasta do port>/user` (modo portátil, como sempre foi).

O mesmo valor vai para `--user`, `ISS_CONFIG` (`tlg.ini`), `TLG_GPU_USER_DIR`/cache de shaders, backups e lista de saves, nos launchers Qt e Tk e em `run.py`. Sugestões estáveis: `Documentos\inFAMOUS-PC` (Windows) e `~/.local/share/infamous-pc` (Linux).

**Mudar a pasta** copia, nunca move: se a antiga tem saves e a nova está vazia, o launcher pergunta "Copiar os saves e configurações para a nova pasta?" (saves, backups e `tlg.ini`; o cache de shaders é refeito). Confere espaço, copia para um nome temporário e renomeia; nada é apagado da pasta antiga. Com `ISS_USER_DIR` definida a escolha fica travada na tela.

**Saves de outras instalações:** uma varredura limitada (profundidade, número de pastas e 8 s) em Downloads, Área de Trabalho, Documentos, home, `C:\Games`, `D:\`..., pastas irmãs e pais da atual, incluindo pasta-dentro-de-pasta (`infamous-windows-x86_64\infamous-windows-x86_64\user`). Mostra "Encontramos saves em ... Importar?" na tela e um aviso no Início quando a pasta atual não tem saves.

**Lista de saves:** um cartão por slot (`infamousssAuto0/1`, `infamousssUser0..N`) com o título de `param.bin` (subtítulo: capítulo, karma, %), data, tamanho e a miniatura `icon0.png` do próprio save (lida na hora; nada é copiado para o repositório). Ações: exportar `.zip`, duplicar (para o próximo `infamousssUserN`), excluir. Barra superior: **Fazer backup agora** (todos os saves), importar `.zip`, exportar tudo.

**Backups:** lista de `savedata_backups` (data, quantidade de saves, tamanho) com **Restaurar**; manter os últimos N (padrão 10) e **pasta extra de backup** (outro disco) que recebe uma cópia de cada backup. Um backup é feito antes de toda partida (pulado se nada mudou desde o último).

**Segurança** (`launcher/savemanager.py`, testes em `test_savemanager.py`, `test_userdata.py`, `test_qt_saves.py`):

- toda operação que pode substituir ou apagar um save passa por `guarded()`/`backup_now()`: o jogo não pode estar rodando (processo `tlg-probe`), o backup de segurança é feito antes (e copiado para a pasta extra); se o backup falhar, nada muda;
- copiar para temporário e renomear; troca de slot com desfazer em caso de erro; restauração troca por renomeações;
- `.zip` validado antes de escrever qualquer byte: estrutura `savedata/<usuario>/CUSA00223/<slot>[.sce_sys]/...` (até 2 pastas de embrulho), recusa total de `../`, caminho absoluto, `C:`, links e tamanhos absurdos (5000 arquivos / 1 GiB); nunca sobrescreve sem perguntar (a tela lista "novo" e "substitui seu ... por ...");
- erros (permissão, disco cheio) viram mensagens claras em pt-BR/en-US; com o jogo aberto as ações que escrevem em `savedata` ficam desativadas.

Formato do `param.bin` lido: `title[128]`, `subtitle[128]`, `detail[1024]`, tempo de 64 bits em `0x508` (também lê PSF `param.sfo`).

## Gráficos

Entrada **Gráficos** (azul `#4d8bff`). Reúne, sem chave nova, o que o launcher Tk (caixa Vídeo) e as abas Imagem/Desempenho/Overlay do menu Insert já tinham. Lê e grava pelo mesmo caminho das outras telas: `settings.json` (modo de janela, VSync, mostrar FPS, precisão da GPU 0/1/2, leitura antecipada, escrita sem esperar, orçamento de VRAM) e `tlg.ini` via `Backend.setFix` (a mesma tabela de `backports_table.py` da tela Correções: FSR 1, nitidez RCAS e intensidade, limite de FPS, quadros na fila, compilação assíncrona, leituras adiadas/atrasadas, pipelines em segundo plano, itens do overlay). Cada opção tem a etiqueta **no jogo** (muda também pelo Insert, na hora) ou **reinicia o jogo**.

- **Predefinições** (`qt/gfxlogic.py`, só mexem em VSync, limite de FPS, quadros na fila, FSR 1, RCAS, compilação assíncrona, leituras adiadas/atrasadas e pipelines em segundo plano; modo de janela, VRAM, precisão da GPU e overlay ficam como estão): **Desempenho** (VSync desligado, 2 quadros na fila, FSR 1), **Equilibrado** (tudo no padrão da tabela; é o "voltar ao padrão" dessas chaves), **Qualidade** (FSR 1, leituras em dia), **Fiel ao PS4** (30 FPS, sem FSR nem nitidez, leituras em dia). A tela diz qual predefinição vale agora ou "personalizada". Duas predefinições **completas** vêm antes das quatro: **Recomendada atualmente** (a configuração do autor: precisão Preciso, leitura adiantada desligada, FSR 1, BP07/BP08; distância de visão e LOD sempre no nativo) e **Placas AMD** (a mesma, com a reconvergência máxima BP06 e o cache de pipelines do driver BP08 desligados e os contornos de imagem 1D ligados; não corrige o crash da RX 580, que ainda não tem correção). Elas definem também a precisão da GPU, o que as quatro simples não fazem, e só mexem nas chaves que listam (`presets_current.py`, `presets_amd.py`).
- **Exportar e importar configuração** (`preset_share.py`, sem Qt; os mesmos botões ficam na janela de correções do Tk): grava as opções de gráficos, os BPnn, as chaves A/B e as opções do menu do jogo num `.iss-preset.json` (`{"format": "iss-preset", "version": 1, "game": "CUSA00223", "name", "what", "exported", "port_version", "settings", "items"}`). Não leva orçamento de VRAM, modo de janela, idiomas, pastas, saves nem as correções AMD em Automático. Ao importar, o launcher valida o arquivo (formato, versão, jogo, ids desconhecidos ignorados, valores fora da faixa recusados), mostra quantas opções mudam e, ao confirmar, aplica como uma predefinição completa (`gfxlogic.custom_targets` e `apply_targets`); o perfil aparece como personalizado com o nome do arquivo.
- **VRAM:** "Sua GPU (nome) tem X GB de VRAM. Orçamento sugerido: N MB", lido numa thread por `nvidia-smi`, depois `/sys/class/drm/card*/device/mem_info_vram_total` (amdgpu) e por fim os heaps `DEVICE_LOCAL` do `vulkaninfo`. A regra é a do Tk (6 GB: 3500, 8: 5000, 12: 8500, 16: 12000, 24: 19000; entre os pontos, interpolação), com botão "Usar N MB" e aviso quando o orçamento passa de 90% da placa. Sem leitura, mostra o texto da regra do Tk.
- **Cache de shaders:** tamanho (`shader_cache_size`) e **Limpar cache** (`clear_shader_cache`) com confirmação, desativado com o jogo aberto; os textos são os do Tk.
- **Não incluído:** resolução de saída, upscaler temporal (FSR 3/4, TAA, DLSS) e resolução de renderização não existem no inFAMOUS (o menu do jogo só tem FSR 1 e RCAS; ver o comentário em `tlg_overlay.cpp`), então não há controle para eles.

## Controles

Entrada **Controles** (néon `#ff3d9a`). Lógica e nomes de tecla em `launcher/padsettings.py`, conferidos com o código do jogo (`port/src/runtime_pad.c`, `port/gpu/shim/tlg_settings.cpp`) por `test_qt_screens_a.py`.

- **Controle:** nome e tipo detectados (o mesmo detector do chip do Início), zona morta dos analógicos (0-40%, `stick_deadzone`, reinicia) e velocidade da mira do grafite (`sensor_aim`, no jogo). Texto curto sobre sensor/touchpad e botão para a tela **Guias de controle** (só aparece se essa tela existir).
- **Mouse:** `mouse_enabled`, `mouse_sensitivity`, `mouse_invert_y`, `mouse_smoothing`, `mouse_accel`, `mouse_deadzone` no `tlg.ini` (a chave sai quando volta ao padrão) e o botão **Desligar o mouse** (`mouse_enabled=0`, o mesmo que `TLG_MOUSE=0`). Modo touchpad do mouse: `settings.json` (`mouse_pad`, `mouse_pad_key`, `mouse_pad_mode`, `mouse_pad_sens`, `mouse_pad_drag`) virando `ISS_MOUSE_PAD`, `ISS_MOUSE_PAD_KEY` (Ctrl esquerdo ou X1 por padrão), `ISS_MOUSE_PAD_MODE` (`hold`/`toggle`), `ISS_MOUSE_PAD_SENS` e `ISS_MOUSE_PAD_DRAG` em `build_env`, só quando diferem do padrão. O mouse como câmera e como touchpad já está no jogo; ainda foi pouco testado por outros jogadores.
- **Teclado:** tabela do mapa padrão (lida do `key_map` de `runtime_pad.c`; um teste compara) e o campo avançado do `TLG_KEY_MAP` (`botão=tecla`, vírgulas, `|` para várias fontes, `none`, até 1023 caracteres e 8 fontes por botão; botões e nomes de mouse como em `apply_button_map`). Erros de formato bloqueiam o botão Salvar; nome de tecla que a lista do launcher não conhece é só aviso (quem decide é o SDL no jogo). O valor vai em `settings.json` (`key_map`) e sai como `ISS_KEY_MAP`; um `TLG_KEY_MAP`/`ISS_KEY_MAP` do sistema vale mais.

## Suporte e Atualizações

**Suporte** (ciano) tem três blocos. *Problemas conhecidos para a sua GPU*: tabela local em `qt/data/known_issues.json` (GPU/fabricante/driver/SO/VRAM e o que fazer; as que casam com a GPU detectada ficam no topo, as outras atrás de um botão). *Relatar um problema*, em 3 passos: (1) coleta local: GPU, driver e VRAM (`vulkaninfo`, com o mesmo ambiente do jogo), SO, CPU com SSE4A/AVX2/F16C (`/proc/cpuinfo`; no Windows a sonda `tlg-probe --cpu-features`), RAM, versão do port (arquivo `VERSION` do pacote, ou o commit da linha `Build:` do último log), jogo (TITLE_ID e APP_VER do `param.sfo`); (2) log: lista `out/logs/run-*.log` (o mais recente vem marcado), mostra o resumo (`Build:`, bloco `Config:`, `Fault:`/`STOP:`/`FREEZE:` e FPS médio) e faz a cópia **redigida**; (3) saída: *Copiar relatório* (markdown), *Salvar .zip* (relatório + log redigido) e *Abrir issue no GitHub* (`webbrowser.open` em `.../issues/new?title=&body=`; corpo acima de 6000 caracteres, ou URL acima de 7500, leva só o resumo e pede para anexar o .zip). Links: Discord, YouTube, GitHub e a pasta de logs.

Redação (`qt/report.py`): o nome em `C:\Users\<nome>`, `/home/<nome>`, `/Users`, `/var/home`, `/run/media` e `/media` vira `<user>`, mais o login/home/hostname desta máquina onde aparecerem soltos; o resto do texto fica igual. O relatório e o log são conferidos de novo antes de gravar o .zip (`leftover_home_paths`): se sobrar caminho pessoal, nada é gravado. **Nenhuma chamada de rede** e nada é enviado sem o clique do jogador.

**Atualizações** (amarelo): versão instalada, *Verificar atualizações* (um único GET HTTPS em `api.github.com/repos/HayatoG/iss_pc/releases/latest`, timeout de 10 s, só o `User-Agent` `iss_pc-launcher/<versão>`; sem dados do jogador), notas da release no idioma da interface (o corpo que o CI gera tem `## Novidades` e `## What's new`; blocos `<!-- notes:pt-BR -->` também são lidos) e *Página da release*, que abre a release no navegador. *Verificar ao abrir* fica **desligado** por padrão (`check_updates_on_open` no `settings.json`). O ponto amarelo do trilho aparece só quando a última versão conhecida (`update_seen` no settings) é mais nova que a instalada. Erros de rede viram mensagens simples (sem internet, demora, limite do GitHub, resposta estranha).

*Baixar e instalar* (`qt/updater.py`, só no pacote; numa cópia do git a tela mostra uma dica de `git pull` + `build.sh`): pega na resposta da mesma API o asset desta plataforma (`infamous-linux-x86_64.tar.gz` ou `infamous-windows-x86_64.zip`) e, se a release tiver, o `SHA256SUMS` (gerado pelo passo Publish do `.github/workflows/build.yml`). Três confirmações, cada uma num cartão: (1) baixar: tamanho, pasta nova, o que fica; o download roda fora da thread da interface, com barra de progresso e *Cancelar*, só em HTTPS (também nos redirecionamentos) e só de `github.com/HayatoG/iss_pc/releases/download/`, para um arquivo temporário ao lado da instalação; o tamanho tem de ser exatamente o da API e o SHA-256 o do `SHA256SUMS` (sem ele, só o tamanho, e o cartão diz isso); arquivo errado é apagado; (2) instalar: extrai numa pasta **nova** irmã (`<pacote>-<tag>`, `-2`... se já existir) por uma pasta temporária oculta, recusando membros com caminho absoluto, `..` ou link para fora, e confere que o launcher está lá; a pasta em uso nunca é sobrescrita (no Windows o .exe aberto não pode ser trocado); se a pasta de dados está dentro da instalação atual (modo portátil, `user/`), ela é copiada para a nova, menos o cache de shaders; se está fora (`ISS_USER_DIR` ou o arquivo de ponteiro do `userdata.py`), a versão nova usa a mesma; o `settings.json` já fica fora; (3) reiniciar: abre o launcher da pasta nova (destacado, com `PYINSTALLER_RESET_ENVIRONMENT=1` e o `LD_LIBRARY_PATH` original) e fecha este; *Depois* deixa o botão *Abrir a versão nova*. A pasta antiga fica intacta: para voltar, basta abrir o launcher dela. Testes: `port/tests/test_updater.py` (sem rede: downloads por um `opener` falso).

Versão: o pacote agora leva um arquivo `VERSION` com o nome da release (`vAAAA.MM.DD-<commit>`, escrito por `packaging/*/package.sh` quando `GITHUB_SHA` existe). Sem ele, só o commit do `Build:` do log é conhecido e a comparação diz "é outra versão" em vez de "mais nova". Testes: `port/tests/test_qt_support.py` (sem rede: o GET usa um `opener` de teste).

## Avançado

A tela **Avançado** (última do menu) reúne os parâmetros de ajuste fino da GPU e do runtime que a tela Correções não cobre: sincronização e tempo, threads e filas, leituras da GPU e memória, shaders e pipelines, contornos de imagem e diagnóstico (logs). Cada parâmetro é um interruptor, um número com unidade e faixa válida, ou uma lista; onde o jogo decide sozinho aparece "Automático".

- **Só o que você mudar vale.** Um valor igual ao padrão não é guardado. Os valores alterados ficam no `tlg.ini` (`adv_<nome>`), viram `ISS_<NOME>` na partida e aparecem no bloco `Config:` do log. Uma variável já definida fora do launcher continua vencendo.
- **Busca, restaurar padrões e copiar minha combinação** (as variáveis alteradas, para colar numa issue).
- **São ajustes para teste.** Um valor errado pode derrubar o FPS, quebrar um efeito ou impedir o jogo de abrir; restaurar os padrões desfaz tudo desta tela. Os rótulos "Experimental" marcam os de maior risco.
- A tabela vive em `port/launcher/advanced_table.py` (uma linha por parâmetro, textos em PT-BR e EN-US) e o teste `tests/test_backports_table.py` confere que cada nome existe no código.

Dois parâmetros novos merecem destaque: `ASYNC_WAIT_MS` (espera curta por um shader ou pipeline em compilação, em vez de pular o desenho; 0 = desligado, o padrão) e `DRIVER_CACHE_SAVE_S` (de quanto em quanto tempo o cache do driver é gravado; 60 s, antes 180 s).

## Estrutura

- `qt/qml/Launcher/Theme.qml`: todas as cores, tamanhos e ícones (um lugar só).
- `qt/logic.py`: lógica sem Qt (textos, chips, novidades, save, grupos de correções); coberta por `port/tests/test_qt_launcher.py`.
- `qt/backend.py`: o objeto que o QML usa (`Backend`).
- `qt/graphics.py`, `qt/controls.py`, `qt/gfxlogic.py`, `launcher/padsettings.py`, `qt/rowsmodel.py`: telas Gráficos e Controles (controladores, lógica sem Qt, modelo de linhas); componentes `OptionRow`, `LevelSlider`, `SectionTitle` e `NoteBar` em `qml/Launcher/`.
- `qt/i18n/strings.json`: textos novos em pt-BR e en-US; textos que o Tk já tem (`exp_live`, `exp_recommended`...) são reaproveitados de `infamous_launcher.TEXT`.

## Capturas sem abrir o jogo

```bash
cd port/launcher
QT_QPA_PLATFORM=offscreen QT_QUICK_BACKEND=software python3 -m qt.screenshots ../../shots
```

Usa dados de demonstração e uma pasta de configuração temporária; não toca nos seus saves nem no `settings.json`.

## Guias de controle

Entrada **Guias de controle** (destaque néon). Chips Xbox / Nintendo Switch Pro / 8BitDo (pré-seleciona o controle detectado; Xbox se não houver), a imagem-guia no idioma do launcher (pt-BR ou en-US) ajustada à largura da janela com rolagem e zoom (+/−/Ajustar), **Salvar imagem…** (PNG), **Copiar imagem** (área de transferência, para colar no Discord) e **Abrir pasta das imagens**. Texto curto explicando que o port simula giroscópio e touchpad e um link para a seção do README sobre teclado e mouse.

As 6 imagens ficam em `port/launcher/qt/assets/controller_guides/inFAMOUS-controle-{xbox,switch,8bitdo}-{pt-BR,en-US}.png` e são lidas em tempo de execução; se faltarem, a tela mostra "Imagens não incluídas neste pacote" (dá para trocar por ilustrações próprias com os mesmos nomes). Ao empacotar (PyInstaller), inclua `qt/assets` como dados, junto de `qt/qml`, `qt/fonts` e `qt/i18n`. O gerador é `port/tools/controller_guides/make_guides.py` (README ao lado; as fotos-base não ficam no repositório).
