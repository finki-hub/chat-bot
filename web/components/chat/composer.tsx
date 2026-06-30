import {
  type KeyboardEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';

import {
  ComposerActions,
  type ComposerStatus,
} from '@/components/chat/composer-actions';
import { t } from '@/lib/i18n';
import { groupModelsByProvider } from '@/lib/use-models';

export type ComposerProps = {
  disabled?: boolean;
  model: string;
  models: string[];
  modelsError?: boolean;
  modelsLoading?: boolean;
  onModelChange: (model: string) => void;
  onReasoningChange: (reasoning: boolean) => void;
  onStop: () => void;
  onSubmit: (text: string) => void;
  reasoning: boolean;
  status: ComposerStatus;
};

export const Composer = ({
  disabled,
  model,
  models,
  modelsError,
  modelsLoading,
  onModelChange,
  onReasoningChange,
  onStop,
  onSubmit,
  reasoning,
  status,
}: ComposerProps) => {
  const [value, setValue] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const isBusy = status === 'streaming' || status === 'submitted';
  const groups = useMemo(() => groupModelsByProvider(models), [models]);
  const noModels = models.length === 0;
  const modelSelectDisabled =
    (disabled ?? false) || modelsLoading === true || noModels;
  let modelPlaceholder = t('composer.model');
  if (modelsLoading === true) {
    modelPlaceholder = t('composer.modelsLoading');
  } else if (modelsError === true) {
    modelPlaceholder = t('composer.modelsError');
  }

  // Keep the input ready for typing on desktop: focus on mount and whenever a
  // response finishes (status returns to `ready`). Skip on small screens (so the
  // mobile keyboard does not pop up) and while a modal is open.
  const focusInput = useCallback(() => {
    if (document.querySelector('[role="dialog"]') !== null) {
      return;
    }
    if (
      typeof matchMedia === 'function' &&
      matchMedia('(min-width: 768px)').matches
    ) {
      textareaRef.current?.focus();
    }
  }, []);

  useEffect(() => {
    if (status === 'ready') {
      focusInput();
    }
  }, [status, focusInput]);

  // Start typing anywhere on the page to focus the composer, so the first
  // keystroke lands in the input even when it was never focused. Skip shortcuts
  // and non-character keys, and stay out of the way of other editable fields.
  useEffect(() => {
    const onTypeAhead = (e: globalThis.KeyboardEvent) => {
      if (
        e.defaultPrevented ||
        e.ctrlKey ||
        e.metaKey ||
        e.altKey ||
        e.key.length !== 1
      ) {
        return;
      }
      const active = document.activeElement;
      const isEditable =
        active instanceof HTMLElement &&
        (active.isContentEditable ||
          active.tagName === 'INPUT' ||
          active.tagName === 'TEXTAREA' ||
          active.tagName === 'SELECT');
      if (isEditable || document.querySelector('[role="dialog"]') !== null) {
        return;
      }
      textareaRef.current?.focus();
    };

    document.addEventListener('keydown', onTypeAhead);
    return () => {
      document.removeEventListener('keydown', onTypeAhead);
    };
  }, []);

  const submit = () => {
    const trimmed = value.trim();
    if (!trimmed || isBusy || disabled) {
      return;
    }
    onSubmit(trimmed);
    setValue('');
    focusInput();
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

  return (
    <div className="bg-background px-3 pb-3 pt-2 sm:px-4">
      <div className="mx-auto w-full max-w-3xl">
        <div className="flex flex-col rounded-3xl border border-input bg-card shadow-lg shadow-black/5 transition-[color,box-shadow] focus-within:border-foreground/25 focus-within:ring-2 focus-within:ring-foreground/[0.06] dark:shadow-black/25">
          <textarea
            aria-label={t('composer.message')}
            autoComplete="off"
            className="ph-no-capture field-sizing-content max-h-48 min-h-[52px] w-full resize-none bg-transparent px-4 pt-3 pb-2 text-sm outline-none placeholder:text-muted-foreground disabled:opacity-50"
            data-testid="composer-input"
            disabled={disabled}
            name="message"
            onChange={(e) => {
              setValue(e.target.value);
            }}
            onKeyDown={onKeyDown}
            placeholder={t('composer.placeholder')}
            ref={textareaRef}
            rows={1}
            value={value}
          />
          <ComposerActions
            disabled={disabled}
            groups={groups}
            isBusy={isBusy}
            model={model}
            modelPlaceholder={modelPlaceholder}
            modelSelectDisabled={modelSelectDisabled}
            modelsLoading={modelsLoading}
            onButtonClick={onButtonClick}
            onModelChange={onModelChange}
            onReasoningChange={onReasoningChange}
            reasoning={reasoning}
            status={status}
            submitDisabled={
              isBusy ? false : (disabled ?? false) || value.trim().length === 0
            }
          />
        </div>
        <p className="mt-2 text-center text-xs text-muted-foreground">
          {t('composer.disclaimer')}
        </p>
      </div>
    </div>
  );
};
