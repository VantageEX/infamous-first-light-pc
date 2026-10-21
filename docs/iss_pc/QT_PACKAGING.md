# Pacotes com o launcher Qt

A partir desta versão os pacotes (`infamous-linux-x86_64.tar.gz`, `infamous-windows-x86_64.zip`)
abrem o launcher **novo (Qt Quick)** por padrão. O launcher antigo (Tk) continua no pacote como reserva.

| Executável | Launcher |
|---|---|
| `inFAMOUS Second Son` (`.exe` no Windows) | Qt Quick, o padrão (atalho do menu, `install_desktop.sh`) |
| `inFAMOUS Second Son (classic)` | Tk, a reserva: use se o novo não abrir na sua máquina |

Os dois compartilham as bibliotecas (uma pasta só), as mesmas configurações, os mesmos saves e o mesmo
caminho de início do jogo (`infamous_run`). `--play`, `--script` e `--clear-shader-cache` funcionam nos dois.

## Como é gerado

- `port/packaging/launcher.spec` (PyInstaller) congela os dois launchers; `packaging/linux/package.sh` e
  `packaging/windows/package.sh` só instalam as dependências no `out/pyenv` (PyInstaller, iced-x86 e
  `PORT_QT_PACKAGE`, hoje `PySide6-Essentials==6.12.0`) e chamam o spec. Variáveis em `packaging/port.env`:
  `PORT_LAUNCHER` (Qt), `PORT_LAUNCHER_CLASSIC` (Tk), `PORT_QT_PACKAGE`.
- Dados do Qt (QML, `fonts/`, `i18n/`, e qualquer pasta nova de arquivos que não sejam código em `launcher/qt/`,
  como `assets/`) entram sozinhos em `qt/...` dentro do pacote; `qt/app.py` os acha por `Path(__file__).parent`,
  que no executável congelado aponta para esse mesmo lugar. O ícone da janela vai em `infamous.png` na raiz.
- O pacote leva um arquivo `VERSION` (`vAAAA.MM.DD-<commit>`), escrito por `package.sh` quando `GITHUB_SHA` existe (o CI); é com ele que a tela Atualizações sabe a versão instalada.
- Redução de tamanho: módulos PySide6 que o launcher não usa são excluídos (WebEngine, Multimedia, 3D,
  Quick3D, Pdf, Sql, Charts, Widgets, ...) e o spec descarta arquivos que o launcher nunca carrega: estilos
  do Qt Quick Controls que não são o Basic, traduções do Qt, plugins de depuração QML, plataformas para
  dispositivos embarcados, o tema GTK3 (e as dezenas de bibliotecas GTK que ele arrasta no Linux),
  Wayland compositor, QtTest. Permanecem as plataformas `xcb`/`wayland`/`offscreen`/`minimal` (Linux) e
  `windows`/`offscreen`/`minimal` (Windows) e os formatos de imagem PNG/JPEG/SVG/GIF/ICO.

## Autoteste

`"inFAMOUS Second Son" --selftest` (Qt) sobe o QML na plataforma `offscreen` com o renderizador de
software, confere QML, fontes, ícone e traduções (pt-BR e en-US completas), visita todas as telas,
renderiza um quadro e imprime `selftest ok` (código 0). Se faltar algo, imprime `selftest FAILED: ...`
e sai com código diferente de 0. Não lê as configurações nem os saves do jogador (usa uma pasta
temporária), não inicia o jogo. Como o `.exe` do Windows é "windowed" (sem console), o resultado
também é gravado no arquivo indicado por `ISS_SELFTEST_OUT`.

O CI roda esse teste logo depois do passo Package nos jobs `linux` e `windows` (passo "Smoke test");
um launcher que não abre derruba o job antes da release.

## Tamanho (medido no Linux, Python 3.14, PySide6-Essentials 6.12.0)

| | pasta do launcher | tar.gz da pasta |
|---|---|---|
| só o Tk (antes) | 38 MB | 17 MB |
| Qt + Tk, sem enxugar | 251 MB | n/d |
| Qt + Tk, como entra no pacote | 186 MB | 75 MB |

O `libicudata` (31 MB) e as bibliotecas Qt Core/Gui/Quick/Qml/Widgets (~45 MB) são o piso. O pacote
final soma a isso o jogo nativo (`out/`), então espere cerca de 55 MB a mais no download por causa do Qt.
Windows (estimado, não medido): pasta do launcher ~140-170 MB, zip ~55-70 MB a mais que o launcher Tk.
