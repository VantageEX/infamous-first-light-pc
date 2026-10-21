# Teste local: main + PR #227

- Base main: `48022f0` (inclui RX580 #232/#278 e apoio #284).
- PR #227: `cba0486c40f3d24040b44af33e60baecbbe9c3a4`.
- Branch: `feat/test-pr227-main`, worktree `~/decomp/infamous_pc-pr227`.
- Integração local para teste; não altera a main nem publica release.

Conflitos resolvidos preservando os diagnósticos de performance da main e as mudanças do PR. No GC em lote, cada imagem é liberada uma única vez e mantém o motivo de descarte nos eventos de diagnóstico. Cache SPIR-V é carregado em todos os caminhos conforme o PR, preservando eventos de tradução/leitura. Readback local permanece opt-in. Os gatilhos e condições de release manual da main foram conferidos.

## Roteiro

1. Abrir o launcher desta branch e iniciar manualmente com as opções atuais do jogador; as opções experimentais novas continuam nos padrões do PR.
2. Comparar o mesmo save/cena com a main: FPS, engasgos, fumaça, partículas, luzes, água e fogo. A alteração do tradutor renova a chave de cache; aguardar a compilação inicial antes de comparar desempenho.
3. Conferir idioma automático e opções detectadas no dump, saves, mods e fechamento normal.
4. Testar VSync e redimensionamento; não instalar atualização pelo launcher durante este teste da branch.

Compilação, testes locais e resultados de gameplay são registrados em `~/decomp/iss_pr227_review/` e nas docs de estado do workspace. O usuário faz a validação visual; nenhum resultado de gameplay é presumido.

## Validação local em 19/10/2026

Build completo Linux com `ISS_LTO=OFF ISS_PGO=off` passou, assim como a suíte C de `build.sh --test-only`, os executáveis `recompiler-fixes-test` (padrões e LDS/WQM desligados), `barrier-merge-test`, `readback-queue-test` e `async-pipeline-cache-test` na RTX2060. Source Qt `--selftest` passou. Python: 601 testes, 84 ignorados e dois erros antigos de importação nos testes de GoW3 (`tlg_launcher_win`, `tlg_assets` ausentes), sem outras falhas. Os arquivos antigos não foram removidos.

O merge textual deixou duas referências ao campo removido `lazy_programs` nos diagnósticos. Corrigidas mantendo eventos de pedidos e cargas reais do cache. Os binários foram compilados/testados com esse ajuste antes do commit final; o identificador do build pode mostrar o commit de integração com `+dirty`.

### Limite da hipótese de ballot

Conferido no backend desta branch: `EmitLoadSharedU64` já calcula o índice com `offset >> 3`. A nova máscara explícita `offset & ~7u` não muda esse índice. O teste novo do PR verifica a forma do IR, não uma diferença no SPIR-V emitido. Manter a mudança na branch permite testar o conjunto solicitado, mas este trecho sozinho não comprova uma correção de piscar. Há outras mudanças no PR que podem alterar a imagem/desempenho; a validação é do usuário jogando.
