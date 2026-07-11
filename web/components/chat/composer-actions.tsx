import { ArrowUp, Brain, Loader2, Sparkles, Square } from 'lucide-react';

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
import { type ModelGroup, providerLabel, tierLabel } from '@/lib/model-catalog';
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

  const selectedModel = groups
    .flatMap((group) => group.providers)
    .flatMap((provider) => provider.models)
    .find((entry) => entry.id === model);
  const triggerLabel = selectedModel?.name ?? modelPlaceholder;

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
            <SelectValue placeholder={modelPlaceholder}>
              {triggerLabel}
            </SelectValue>
          </SelectTrigger>
          <SelectContent
            align="end"
            position="popper"
          >
            {groups.map((group) => (
              <SelectGroup key={group.tier}>
                <SelectLabel data-testid="model-tier-label">
                  {tierLabel(group.tier)}
                </SelectLabel>
                {group.providers.map((providerGroup) => (
                  <fieldset
                    aria-label={providerLabel(providerGroup.provider)}
                    className="m-0 border-0 p-0"
                    key={providerGroup.provider}
                  >
                    <span
                      aria-hidden="true"
                      className="block px-2 py-1 text-[10px] font-medium uppercase tracking-wide text-muted-foreground"
                      data-testid="model-provider-label"
                    >
                      {providerLabel(providerGroup.provider)}
                    </span>
                    {providerGroup.models.map((entry) => (
                      <SelectItem
                        key={entry.id}
                        textValue={entry.name}
                        value={entry.id}
                      >
                        {entry.name}
                      </SelectItem>
                    ))}
                  </fieldset>
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
