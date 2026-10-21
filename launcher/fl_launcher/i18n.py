# SPDX-License-Identifier: GPL-2.0-or-later
"""Launcher interface texts (Russian and English)."""
STRINGS = {
    'window_title': {'ru': 'inFAMOUS First Light — PC-порт', 'en': 'inFAMOUS First Light — PC port'},
    'subtitle': {'ru': 'Неофициальный нативный порт. Нужна ваша собственная копия игры.',
                 'en': 'Unofficial native port. You need your own copy of the game.'},
    'game_folder': {'ru': 'Папка с игрой', 'en': 'Game folder'},
    'no_game': {'ru': 'Выберите папку с дампом игры (в ней должны быть eboot.bin и sce_sys).',
                'en': 'Pick your game dump folder (it holds eboot.bin and sce_sys).'},
    'no_eboot': {'ru': 'В этой папке нет eboot.bin.', 'en': 'This folder has no eboot.bin.'},
    'no_sfo': {'ru': 'В этой папке нет sce_sys/param.sfo.', 'en': 'This folder has no sce_sys/param.sfo.'},
    'bad_sfo': {'ru': 'Не удалось прочитать param.sfo: {error}', 'en': 'Could not read param.sfo: {error}'},
    'game_untested': {'ru': 'Эта версия ({title_id}) ещё не проверялась с портом.',
                      'en': 'This version ({title_id}) has not been tested with the port yet.'},
    'version_text': {'ru': 'версия {version}', 'en': 'version {version}'},
    'choose_folder': {'ru': 'Выбрать папку с игрой', 'en': 'Choose game folder'},
    'change_folder': {'ru': 'Сменить папку с игрой', 'en': 'Change game folder'},
    'game_language': {'ru': 'Язык игры', 'en': 'Game language'},
    'language_auto': {'ru': 'Как в системе', 'en': 'System language'},
    'ui_language': {'ru': 'Язык лаунчера', 'en': 'Launcher language'},
    'auto': {'ru': 'Авто', 'en': 'Auto'},
    'language': {'ru': 'Язык', 'en': 'Language'},
    'display': {'ru': 'Экран', 'en': 'Display'},
    'display_mode': {'ru': 'Режим окна', 'en': 'Window mode'},
    'renderer': {'ru': 'Рендер', 'en': 'Renderer'},
    'renderer_classic': {'ru': 'Обычный — запасной', 'en': 'Classic — fallback'},
    'renderer_v2': {'ru': 'v2 — быстрее (рекомендуется)', 'en': 'v2 — faster (recommended)'},
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
    'advanced': {'ru': 'Дополнительно', 'en': 'Advanced'},
    'advanced_on': {'ru': 'Передавать переменные ядру', 'en': 'Pass the variables to the core'},
    'advanced_note': {'ru': 'Для тестов: они дополняют и заменяют настройки ядра, с которыми запускается игра.',
                      'en': 'For testing: they add to and override the core switches the game runs with.'},
    'extra_env_hint': {'ru': 'Переменные ядра, по одной на строку: TLG_OCCLUSION_MODE=2',
                       'en': 'Core variables, one per line: TLG_OCCLUSION_MODE=2'},
    'play': {'ru': 'ИГРАТЬ', 'en': 'PLAY'},
    'stop': {'ru': 'ОСТАНОВИТЬ', 'en': 'STOP'},
    'log': {'ru': 'Журнал', 'en': 'Log'},
    'open_logs': {'ru': 'Папка логов', 'en': 'Logs folder'},
    'open_saves': {'ru': 'Папка сохранений', 'en': 'Saves folder'},
    'create_report': {'ru': 'Собрать отчёт', 'en': 'Create report'},
    'report_done': {'ru': 'Отчёт собран', 'en': 'Report created'},
    'report_failed': {'ru': 'Не удалось собрать отчёт: {error}', 'en': 'Could not create the report: {error}'},
    'test_build': {'ru': 'ТЕСТОВАЯ СБОРКА', 'en': 'TEST BUILD'},
    'test_hint': {'ru': 'После игры нажмите «Собрать отчёт» и отправьте архив разработчику.',
                  'en': 'After playing, press “Create report” and send the archive to the developer.'},
    'starting': {'ru': 'Запуск…', 'en': 'Starting…'},
    'running': {'ru': 'Игра запущена', 'en': 'The game is running'},
    'preparing_shaders': {'ru': 'Подготовка шейдеров, только в первый раз… {percent}%',
                          'en': 'Preparing shaders, first start only… {percent}%'},
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
