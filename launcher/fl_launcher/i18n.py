# SPDX-License-Identifier: GPL-2.0-or-later
"""Launcher interface texts (Russian and English)."""
STRINGS = {
    'window_title': {'ru': 'inFAMOUS First Light — PC-порт', 'en': 'inFAMOUS First Light — PC port'},
    'subtitle': {'ru': 'Неофициальный нативный порт. Нужна ваша собственная копия игры.',
                 'en': 'Unofficial native port. You need your own copy of the game.'},
    'game_folder': {'ru': 'Папка с игрой', 'en': 'Game folder'},
    'browse': {'ru': 'Выбрать…', 'en': 'Browse…'},
    'no_game': {'ru': 'Выберите папку с дампом игры (в ней должны быть eboot.bin и sce_sys).',
                'en': 'Pick your game dump folder (it holds eboot.bin and sce_sys).'},
    'no_eboot': {'ru': 'В этой папке нет eboot.bin.', 'en': 'This folder has no eboot.bin.'},
    'no_sfo': {'ru': 'В этой папке нет sce_sys/param.sfo.', 'en': 'This folder has no sce_sys/param.sfo.'},
    'bad_sfo': {'ru': 'Не удалось прочитать param.sfo: {error}', 'en': 'Could not read param.sfo: {error}'},
    'game_ok': {'ru': '{title} · {title_id} · версия {version}', 'en': '{title} · {title_id} · version {version}'},
    'game_untested': {'ru': 'Эта версия ({title_id}) ещё не проверялась с портом.',
                      'en': 'This version ({title_id}) has not been tested with the port yet.'},
    'game_language': {'ru': 'Язык игры', 'en': 'Game language'},
    'language_auto': {'ru': 'Как в системе', 'en': 'System language'},
    'ui_language': {'ru': 'Язык лаунчера', 'en': 'Launcher language'},
    'display': {'ru': 'Экран', 'en': 'Display'},
    'display_mode': {'ru': 'Режим окна', 'en': 'Window mode'},
    'windowed': {'ru': 'В окне', 'en': 'Windowed'},
    'borderless': {'ru': 'Окно без рамки', 'en': 'Borderless'},
    'fullscreen': {'ru': 'Полный экран', 'en': 'Fullscreen'},
    'vsync': {'ru': 'Вертикальная синхронизация (VSync)', 'en': 'Vertical sync (VSync)'},
    'fps_limit': {'ru': 'Ограничение FPS', 'en': 'Frame rate limit'},
    'unlimited': {'ru': 'Без ограничения', 'en': 'Unlimited'},
    'fsr1': {'ru': 'Масштабирование FSR 1', 'en': 'FSR 1 upscaling'},
    'rcas': {'ru': 'Повышение резкости (RCAS)', 'en': 'Sharpening (RCAS)'},
    'show_fps': {'ru': 'Показывать FPS (F10 в игре)', 'en': 'Show FPS (F10 in game)'},
    'async_shaders': {'ru': 'Фоновая компиляция шейдеров (меньше фризов)',
                      'en': 'Background shader compilation (fewer stutters)'},
    'advanced': {'ru': 'Дополнительно (для тестов)', 'en': 'Advanced (testing)'},
    'extra_env_hint': {'ru': 'Переменные ядра, по одной на строку: TLG_OCCLUSION_MODE=2',
                       'en': 'Core variables, one per line: TLG_OCCLUSION_MODE=2'},
    'play': {'ru': 'ИГРАТЬ', 'en': 'PLAY'},
    'stop': {'ru': 'ОСТАНОВИТЬ', 'en': 'STOP'},
    'log': {'ru': 'Журнал', 'en': 'Log'},
    'open_logs': {'ru': 'Папка логов', 'en': 'Logs folder'},
    'open_saves': {'ru': 'Папка сохранений', 'en': 'Saves folder'},
    'starting': {'ru': 'Запуск…', 'en': 'Starting…'},
    'exited': {'ru': 'Игра завершилась (код {code}).', 'en': 'The game exited (code {code}).'},
    'credits': {'ru': 'Ядро: iss_pc от HayatoG и shadPS4 (GPL-2.0). Не связано с Sony и Sucker Punch.',
                'en': 'Core: iss_pc by HayatoG and shadPS4 (GPL-2.0). Not affiliated with Sony or Sucker Punch.'},
}
UI_LANGUAGES = {'ru': 'Русский', 'en': 'English'}


class Texts:
    def __init__(self, language):
        self.language = language if language in UI_LANGUAGES else 'en'

    def __call__(self, key, **values):
        entry = STRINGS.get(key, {})
        text = entry.get(self.language) or entry.get('en') or key
        return text.format(**values) if values else text
