# Atualização do launcher / Launcher updates

## Português

O launcher Qt consulta a última release publicada do iss_pc. A opção de consultar
ao abrir é configurável e começa desligada; instalar exige confirmação do jogador.
Uma release nova só fica disponível quando o mantenedor a publica manualmente.

No Windows empacotado, o pacote verificado por SHA-256 substitui apenas arquivos
do manifesto `package-files.json`, na pasta atual. Saves, configurações, caches,
mods e arquivos do jogo são preservados. O auxiliar espera o launcher fechar;
o diário permite recuperar uma troca interrompida. O backup da versão anterior
permite restaurá-la até a confirmação da atualização seguinte.

Se a pasta temporária de uma atualização concluída for apagada, o launcher
confere os hashes registrados na instalação antes de descartar o estado antigo.
A restauração deixa de estar disponível. Uma atualização interrompida cuja pasta
temporária foi apagada é reconciliada automaticamente quando todos os arquivos
gerenciados estão na versão anterior ou todos na nova. Se houver mistura de versões,
ou algum arquivo não bater com nenhuma das duas, o arquivo de estado é mantido e a
recuperação exige reinstalar o pacote da release nesta mesma pasta, sem apagar
`user/` nem os saves. Registros antigos sem evidência de conclusão também exigem
intervenção.
A limpeza automática só remove backups concluídos da mesma instalação.

Linux mantém a instalação em pasta vizinha. Checkouts Git não recebem instalação
no lugar pelo updater do Windows.

Validação em 19/10/2026: 46 testes de updater, disco/recuperação e Qt passaram (42 antes da
reconciliação de interrupção, mais 4 dela).
Validado em 19/10/2026 no Windows com testes reais: travas de arquivo, um segundo processo segurando o lock da instalação, espera de processo e o nome tlg-probe.exe; manifesto gerado pelo mesmo comando do empacotamento sobre um subconjunto real do pacote (A→B e restauração exata); e um auxiliar congelado com PyInstaller percorrendo entrega, aplicação, confirmação e restauração com processos reais. Ainda falta: o pacote completo gerado pelo package.sh, o launcher Qt congelado e o jogo. Não houve teste de jogo nesta etapa.

## English

The Qt launcher checks the latest published iss_pc release. Checking on startup
is optional and disabled by default; installation requires player confirmation.
New releases become available when the maintainer publishes them manually.

Packaged Windows builds verify the download using SHA-256 and replace only files
owned by `package-files.json` in the current folder. Game files, saves, settings,
caches and mods are preserved. A helper waits for the launcher to exit, then
applies a recovery journal. The previous version remains restorable until the
next update is confirmed.

If a completed update's temporary folder is deleted, recorded installed hashes
must match before its stale recovery state is retired. Rollback is then
unavailable. An interrupted update whose temporary folder was deleted is
reconciled automatically when every managed file is either all old or all new.
A mixture of versions, or a file matching neither, keeps the state file and needs
a reinstall of the release package into this same folder from the release page,
without deleting `user/` or the saves. Legacy state without completion evidence
still requires intervention. Automatic cleanup only removes completed backups
belonging to this installation.

Linux retains installation into a sibling folder. Windows in-place updates refuse
Git checkouts. On 2026-10-19, all 46 updater, disk recovery and Qt tests passed (42 before the
interrupted-update reconciliation, plus 4 for it).
Validated on 2026-10-19 on Windows with real tests: file locks, a second process holding the install lock, process waits and the tlg-probe.exe name; a manifest from the packaging command on a real package subset (A to B and exact restore); and a PyInstaller-frozen stand-in running handoff, apply, confirm and restore with real processes. Still missing: the full package built by package.sh, the frozen Qt launcher and the game. No game was launched during this work.
