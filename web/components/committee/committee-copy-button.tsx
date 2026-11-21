import { Check, Copy } from 'lucide-react';
import { useEffect, useState } from 'react';

import { Button } from '@/components/ui/button';
import { t } from '@/lib/i18n';

const COPY_FEEDBACK_MS = 2_000;

export const CopyButton = ({
  getText,
  label,
}: {
  getText: () => string;
  label: string;
}) => {
  type CopyState = 'copied' | 'failed' | 'idle';
  const [state, setState] = useState<CopyState>('idle');

  useEffect(() => {
    const timer = setTimeout(() => {
      setState('idle');
    }, COPY_FEEDBACK_MS);
    return () => {
      clearTimeout(timer);
    };
  }, [state]);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(getText());
      setState('copied');
    } catch {
      setState('failed');
    }
  };

  return (
    <Button
      className="pointer-coarse:min-h-11"
      onClick={() => {
        void handleCopy();
      }}
      type="button"
      variant="outline"
    >
      {state === 'copied' ? (
        <Check className="size-4" />
      ) : (
        <Copy className="size-4" />
      )}
      <output>
        {state === 'copied' ? t('committee.copied') : null}
        {state === 'failed' ? t('committee.copyFailed') : null}
        {state === 'idle' ? label : null}
      </output>
    </Button>
  );
};
