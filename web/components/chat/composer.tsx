import { ArrowUp, Loader2, Sparkles, Square } from 'lucide-react';
import { type KeyboardEvent, useState } from 'react';

import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
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
          className="size-4 fill-current"
        />
      );
    }
    return (
      <ArrowUp
        aria-hidden="true"
        className="size-4"
      />
    );
  };

  return (
    <div className="border-t border-border bg-background px-3 py-3 sm:px-4">
      <div className="mx-auto w-full max-w-3xl">
        <div className="flex flex-col rounded-2xl border border-input bg-card shadow-sm transition-[color,box-shadow] focus-within:border-ring focus-within:ring-[3px] focus-within:ring-ring/50">
          <textarea
            aria-label={t('composer.message')}
            className="field-sizing-content max-h-48 min-h-[52px] w-full resize-none bg-transparent px-4 pt-3 pb-2 text-sm outline-none placeholder:text-muted-foreground disabled:opacity-50"
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
          <div className="flex items-center justify-end gap-1.5 px-2 pb-2">
            <Select
              disabled={disabled}
              onValueChange={onModelChange}
              value={model}
            >
              <SelectTrigger
                aria-label={t('composer.model')}
                className="h-8 w-fit max-w-[55%] gap-1.5 rounded-full border-0 bg-transparent px-3 text-xs font-medium text-muted-foreground shadow-none hover:bg-muted hover:text-foreground sm:max-w-[240px]"
                data-testid="composer-model"
                size="sm"
              >
                <Sparkles
                  aria-hidden="true"
                  className="size-3.5"
                />
                <SelectValue placeholder={t('composer.model')} />
              </SelectTrigger>
              <SelectContent
                align="end"
                position="popper"
              >
                {groups.map((g) => (
                  <SelectGroup key={g.provider}>
                    <SelectLabel>{g.provider}</SelectLabel>
                    {g.models.map((id) => (
                      <SelectItem
                        key={id}
                        value={id}
                      >
                        {id}
                      </SelectItem>
                    ))}
                  </SelectGroup>
                ))}
              </SelectContent>
            </Select>
            <Button
              aria-label={isBusy ? t('composer.stop') : t('composer.send')}
              className="size-9 shrink-0 rounded-full"
              data-testid="composer-submit"
              disabled={
                (disabled ?? false) || (!isBusy && value.trim().length === 0)
              }
              onClick={onButtonClick}
              size="icon"
              type="button"
            >
              {renderSubmitIcon()}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};
