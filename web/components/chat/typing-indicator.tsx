import { t } from '@/lib/i18n';

const DOT = 'size-2 animate-bounce rounded-full bg-muted-foreground/60';

export const TypingIndicator = () => (
  <div
    aria-live="polite"
    className="flex items-center gap-1.5 py-1"
    data-testid="typing-indicator"
  >
    <span className="sr-only">{t('thread.thinking')}</span>
    <span className={`${DOT} [animation-delay:-300ms]`} />
    <span className={`${DOT} [animation-delay:-150ms]`} />
    <span className={DOT} />
  </div>
);
