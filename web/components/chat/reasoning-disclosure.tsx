import { ChevronRight } from 'lucide-react';
import { useId, useState } from 'react';

import { MessageResponse } from '@/components/ai-elements/message';
import { t } from '@/lib/i18n';

export const ReasoningDisclosure = ({
  streaming = false,
  text,
}: {
  streaming?: boolean;
  text: string;
}) => {
  // null = follow the stream (open while thinking); a click pins it open or closed.
  const [manualOpen, setManualOpen] = useState<boolean | null>(null);
  const panelId = useId();

  if (text.length === 0) {
    return null;
  }

  const open = manualOpen ?? streaming;

  return (
    <div
      className="mb-2"
      data-testid="reasoning"
    >
      <button
        aria-controls={panelId}
        aria-expanded={open}
        className="inline-flex items-center gap-1 rounded-md text-xs text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        onClick={() => {
          setManualOpen(!open);
        }}
        type="button"
      >
        <ChevronRight
          aria-hidden="true"
          className={`size-3 transition-transform ${open ? 'rotate-90' : ''}`}
        />
        <span className={streaming ? 'animate-pulse' : undefined}>
          {streaming ? t('thread.thinking') : t('thread.reasoning')}
        </span>
      </button>
      {open ? (
        <div
          className="mt-1 border-l-2 border-border pl-3 text-sm text-muted-foreground"
          data-testid="reasoning-panel"
          id={panelId}
        >
          <MessageResponse>{text}</MessageResponse>
        </div>
      ) : null}
    </div>
  );
};
