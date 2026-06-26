import { ChevronRight } from 'lucide-react';
import { useState } from 'react';

import { MessageResponse } from '@/components/ai-elements/message';
import { t } from '@/lib/i18n';

export const ReasoningDisclosure = ({ text }: { text: string }) => {
  const [open, setOpen] = useState(false);

  if (text.length === 0) {
    return null;
  }

  return (
    <div
      className="mb-2"
      data-testid="reasoning"
    >
      <button
        aria-expanded={open}
        className="inline-flex items-center gap-1 text-xs text-muted-foreground transition-colors hover:text-foreground"
        onClick={() => {
          setOpen((v) => !v);
        }}
        type="button"
      >
        <ChevronRight
          aria-hidden="true"
          className={`size-3 transition-transform ${open ? 'rotate-90' : ''}`}
        />
        {t('thread.reasoning')}
      </button>
      {open ? (
        <div className="mt-1 border-l-2 border-border pl-3 text-sm text-muted-foreground">
          <MessageResponse>{text}</MessageResponse>
        </div>
      ) : null}
    </div>
  );
};
