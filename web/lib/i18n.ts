export const messages = {
  'actions.copy': 'Копирај',
  'actions.dislike': 'Не допаѓа',
  'actions.like': 'Допаѓа',
  'actions.regenerate': 'Регенерирај',
  'app.title': 'ФИНКИ Хаб',
  'composer.disclaimer':
    'ФИНКИ Хаб може да направи грешки. Проверете важни информации.',
  'composer.message': 'Порака',
  'composer.model': 'Модел',
  'composer.placeholder': 'Напиши порака…',
  'composer.send': 'Испрати',
  'composer.stop': 'Запри',
  'conversation.delete': 'Избриши',
  'conversation.rename': 'Преименувај',
  'conversation.renamePrompt': 'Ново име на разговорот',
  'error.interrupted': 'Одговорот е прекинат.',
  'error.retry': 'Обиди се повторно',
  'header.github': 'GitHub репозиториум',
  'header.theme': 'Промени тема',
  'header.title': 'ФИНКИ Хаб / Чат',
  'header.toggleSidebar': 'Прикажи/сокриј странична лента',
  'sidebar.history': 'Разговори',
  'sidebar.label': 'Странична лента',
  'sidebar.new': 'Нов разговор',
  'sidebar.toggle': 'Прикажи/сокриј странична лента',
  'thread.emptyDescription': 'Прашај нешто за студиите на ФИНКИ.',
  'thread.emptyTitle': 'Започни разговор',
  'thread.thinking': 'Размислувам…',
} as const;

export type TKey = keyof typeof messages;

export const t = (key: TKey): string => messages[key];
