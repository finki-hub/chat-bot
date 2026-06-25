import { Loader2, Send, Square } from 'lucide-react';
import { type KeyboardEvent, useState } from 'react';

import { t } from '@/lib/i18n';
import { groupModelsByProvider } from '@/lib/use-models';

export type ComposerProps = {
  disabled?: boolean;
  model: string;
  models: string[];
  onModelChange: (model: string) => void;
  onStop: () => void;
  onSubmit: (text: string) => void;
  status: 'error' | 'ready' | 'streaming' | 'submitted';
};

export const Composer = ({
  disabled,
  model,
  models,
  onModelChange,
  onStop,
  onSubmit,
  status,
}: ComposerProps) => {
  const [value, setValue] = useState('');
  const isBusy = status === 'streaming' || status === 'submitted';
  const groups = groupModelsByProvider(models);

  const submit = () => {
    const trimmed = value.trim();
    if (!trimmed || isBusy || disabled) {
      return;
    }
    onSubmit(trimmed);
    setValue('');
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (!(e.key === 'Enter' && !e.shiftKey)) {
      return;
    }

    e.preventDefault();
    submit();
  };

  const onButtonClick = () => {
    if (isBusy) {
      onStop();
    } else {
      submit();
    }
  };

  const renderSubmitIcon = () => {
    if (status === 'submitted') {
      return (
        <Loader2
          aria-hidden="true"
          className="size-4 animate-spin"
        />
      );
    }
    if (isBusy) {
      return (
        <Square
          aria-hidden="true"
          className="size-4"
        />
      );
    }
    return (
      <Send
        aria-hidden="true"
        className="size-4"
      />
    );
  };

  return (
    <div className="flex flex-col gap-2 border-t border-border bg-background p-3">
      <div className="flex items-center gap-2">
        <label
          className="sr-only"
          htmlFor="composer-model"
        >
          {t('composer.model')}
        </label>
        <select
          className="rounded-md border border-border bg-background px-2 py-1 text-sm focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
          data-testid="composer-model"
          disabled={disabled}
          id="composer-model"
          onChange={(e) => {
            onModelChange(e.target.value);
          }}
          value={model}
        >
          {groups.map((g) => (
            <optgroup
              key={g.provider}
              label={g.provider}
            >
              {g.models.map((id) => (
                <option
                  key={id}
                  value={id}
                >
                  {id}
                </option>
              ))}
            </optgroup>
          ))}
        </select>
      </div>
      <div className="flex items-end gap-2">
        <textarea
          aria-label={t('composer.message')}
          className="min-h-[44px] flex-1 resize-none rounded-md border border-border bg-background px-3 py-2 text-sm focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
          data-testid="composer-input"
          disabled={disabled}
          onChange={(e) => {
            setValue(e.target.value);
          }}
          onKeyDown={onKeyDown}
          placeholder={t('composer.placeholder')}
          rows={1}
          value={value}
        />
        <button
          aria-label={isBusy ? t('composer.stop') : t('composer.send')}
          className="inline-flex size-10 items-center justify-center rounded-md bg-primary text-primary-foreground hover:bg-primary/90 focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none disabled:opacity-50"
          data-testid="composer-submit"
          disabled={
            (disabled ?? false) || (!isBusy && value.trim().length === 0)
          }
          onClick={onButtonClick}
          type="button"
        >
          {renderSubmitIcon()}
        </button>
      </div>
    </div>
  );
};
