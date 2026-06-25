// Flat Macedonian-Cyrillic chrome dictionary. Default (and only) locale in v1;
// structured as a flat map so an EN locale can be added later without churn.
export const messages = {
  'actions.copy': 'Копирај',
  'actions.dislike': 'Не допаѓа',
  'actions.like': 'Допаѓа',
  'actions.regenerate': 'Регенерирај',
  'app.title': 'ФИНКИ Хаб',
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
  'sidebar.label': 'Странична лента',
  'sidebar.new': 'Нов разговор',
  'sidebar.toggle': 'Прикажи/сокриј странична лента',
  'thread.emptyDescription': 'Прашај нешто за студиите на ФИНКИ.',
  'thread.emptyTitle': 'Започни разговор',
} as const;

export type TKey = keyof typeof messages;

export const t = (key: TKey): string => messages[key];
