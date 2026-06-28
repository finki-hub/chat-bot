import { ArrowUp, Brain, Loader2, Sparkles, Square } from 'lucide-react';

import type { ModelGroup } from '@/lib/use-models';

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
import { isReasoningCapableModel } from '@/lib/reasoning';

export type ComposerActionsProps = {
  disabled?: boolean;
  groups: ModelGroup[];
  isBusy: boolean;
  model: string;
  modelPlaceholder: string;
  modelSelectDisabled: boolean;
  modelsLoading?: boolean;
  onButtonClick: () => void;
  onModelChange: (model: string) => void;
  onReasoningChange: (reasoning: boolean) => void;
  reasoning: boolean;
  status: ComposerStatus;
  submitDisabled: boolean;
};

export type ComposerStatus = 'error' | 'ready' | 'streaming' | 'submitted';

export const ComposerActions = ({
  disabled,
  groups,
  isBusy,
  model,
  modelPlaceholder,
  modelSelectDisabled,
  modelsLoading,
  onButtonClick,
  onModelChange,
  onReasoningChange,
  reasoning,
  status,
  submitDisabled,
}: ComposerActionsProps) => {
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
    <div className="flex items-center gap-1.5 px-2 pb-2">
      <div
        className="flex min-w-0 flex-1 items-center gap-1.5 overflow-x-auto overscroll-x-contain pb-0.5 sm:justify-end"
        data-testid="composer-chip-scroll"
      >
        <Button
          aria-label={t('composer.reasoning')}
          aria-pressed={reasoning}
          className="h-8 w-fit shrink-0 gap-1.5 rounded-full px-3 text-xs font-medium"
          data-testid="composer-reasoning"
          disabled={(disabled ?? false) || !isReasoningCapableModel(model)}
          onClick={() => {
            onReasoningChange(!reasoning);
          }}
          size="sm"
          type="button"
          variant={reasoning ? 'default' : 'ghost'}
        >
          <Brain
            aria-hidden="true"
            className="size-3.5"
          />
          {t('composer.reasoning')}
        </Button>
        <Select
          disabled={modelSelectDisabled}
          onValueChange={onModelChange}
          value={model}
        >
          <SelectTrigger
            aria-label={t('composer.model')}
            className="h-8 w-fit max-w-[70vw] shrink-0 gap-1.5 rounded-full border-0 bg-transparent px-3 text-xs font-medium text-muted-foreground shadow-none hover:bg-muted hover:text-foreground sm:max-w-[240px]"
            data-testid="composer-model"
            size="sm"
          >
            {modelsLoading === true ? (
              <Loader2
                aria-hidden="true"
                className="size-3.5 animate-spin"
              />
            ) : (
              <Sparkles
                aria-hidden="true"
                className="size-3.5"
              />
            )}
            <SelectValue placeholder={modelPlaceholder} />
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
      </div>
      <Button
        aria-label={isBusy ? t('composer.stop') : t('composer.send')}
        className="size-9 shrink-0 rounded-full transition-transform active:scale-95"
        data-testid="composer-submit"
        disabled={submitDisabled}
        onClick={onButtonClick}
        size="icon"
        type="button"
      >
        {renderSubmitIcon()}
      </Button>
    </div>
  );
};
