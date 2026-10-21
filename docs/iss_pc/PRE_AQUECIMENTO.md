# Pré-aquecimento de shaders: o que sabemos e o que existe

Nota de trabalho (15/10/2026). O jogo traduz os shaders do PS4 (GCN) para Vulkan em tempo de execução; enquanto um shader ou pipeline novo compila em segundo plano, o desenho que dependia dele é pulado. É o chão, os prédios e os efeitos que somem por alguns quadros, principalmente com o cache de shaders frio.

## O que o projeto faz hoje

- **Cache em disco** de pipelines e shaders, com a chave incluindo um hash dos fontes do recompilador, e um aquecimento em segundo plano no boot (quase 6000 pipelines em ~6 s).
- **BP07** (desligado): desenhos importantes compilam na hora em vez de serem pulados.
- **Espera curta** (`ASYNC_WAIT_MS`, desligada): um desenho cujo shader ou pipeline ainda compila espera até N ms (e no máximo `ASYNC_WAIT_FRAME_MS` por quadro) antes de ser pulado. O log mostra a linha `Async wait (total)` a cada 5 s.
- **Cache do driver** salvo a cada 60 s (`DRIVER_CACHE_SAVE_S`).

## O que descobrimos

- O port do God of War III (mesma base) tem o mesmo mecanismo e admite o mesmo problema no README; não há pré-compilação nem ferramenta de aquecimento nele.
- Os shaders do jogo ficam sem compressão nos pacotes `.xpps` de `art/cache` (contêiner `KCAP`, cabeçalho `OrbShdr`). `tools/shader_index.py` indexa todos (10 374 únicos no jogo 1.07), e o hash do índice é o mesmo que o port usa. Tudo só na máquina do jogador; nada do jogo é copiado.
- Uma pré-tradução **completa** sem o jogo rodando não é possível: cada shader vira várias versões conforme o estado do desenho, e esse estado só aparece em jogo.
- O jogo abre pacotes `bNN.xpps` por região, e logo depois pedem-se rajadas de pipelines. O rastreio (`TLG_REGION_TRACE=1`) grava isso em `region_trace.tsv` para estudar o que cada região precisa.

## Próximos passos (propostos)

1. Medir com o rastreio e a espera ligados em áreas novas.
2. Priorizar no aquecimento os pipelines vistos em cada região, no cache local do jogador.
3. Só então, um manifesto de estados por região e tradução sem o jogo (trabalho grande e arriscado).
