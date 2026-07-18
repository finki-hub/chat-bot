import {
  ArrowUp,
  Brain,
  KeyRound,
  Loader2,
  Sparkles,
  Square,
} from 'lucide-react';

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
import { type ModelGroup, providerLabel } from '@/lib/model-catalog';
import { isReasoningCapableModel } from '@/lib/reasoning';

export type ComposerActionsProps = {
  availableProviders: ReadonlySet<string>;
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
  showModelPlaceholder: boolean;
  status: ComposerStatus;
  submitDisabled: boolean;
};

export type ComposerStatus = 'error' | 'ready' | 'streaming' | 'submitted';

export const ComposerActions = ({
  availableProviders,
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
  showModelPlaceholder,
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
    .flatMap((group) => group.models)
    .find((entry) => entry.id === model);
  const triggerLabel = showModelPlaceholder
    ? modelPlaceholder
    : (selectedModel?.name ?? modelPlaceholder);

  return (
    <div className="flex items-center gap-1.5 px-2 pb-2">
      <div
        className="flex min-w-0 flex-1 items-center gap-1.5 overflow-x-auto overscroll-x-contain pb-0.5 sm:justify-end"
        data-testid="composer-chip-scroll"
      >
        <Button
          aria-label={t('composer.reasoning')}
          aria-pressed={reasoning}
          className="min-h-11 w-fit shrink-0 gap-1.5 rounded-full px-3 text-xs font-medium sm:pointer-fine:min-h-8"
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
            className="min-h-11 w-fit max-w-[70vw] shrink-0 gap-1.5 rounded-full border-0 bg-transparent px-3 text-xs font-medium text-muted-foreground shadow-none hover:bg-muted hover:text-foreground sm:max-w-[240px] sm:pointer-fine:min-h-8"
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
            collisionPadding={{ bottom: 12, left: 12, right: 12, top: 56 }}
            position="popper"
          >
            {groups.map((group) => (
              <SelectGroup key={group.provider}>
                <SelectLabel data-testid="model-provider-label">
                  {providerLabel(group.provider)}
                </SelectLabel>
                {group.models.map((entry) => (
                  <SelectItem
                    aria-label={
                      availableProviders.has(entry.provider)
                        ? entry.name
                        : `${entry.name} ${t('composer.apiKeyRequired')}`
                    }
                    className="data-[disabled]:opacity-75"
                    disabled={!availableProviders.has(entry.provider)}
                    key={entry.id}
                    textValue={
                      availableProviders.has(entry.provider)
                        ? entry.name
                        : `${entry.name} ${t('composer.apiKeyRequired')}`
                    }
                    value={entry.id}
                  >
                    <span className="flex min-w-0 flex-1 items-center justify-between gap-3">
                      <span className="truncate">{entry.name}</span>
                      {entry.provider === 'ollama' &&
                      typeof entry.loaded === 'boolean' ? (
                        <span className="flex shrink-0 items-center gap-1 text-xs text-muted-foreground">
                          {entry.loaded
                            ? t('composer.modelLoaded')
                            : t('composer.modelNotLoaded')}
                        </span>
                      ) : null}
                      {availableProviders.has(entry.provider) ? null : (
                        <span className="flex shrink-0 items-center gap-1 text-xs text-muted-foreground">
                          <KeyRound aria-hidden="true" />
                          {t('composer.apiKeyRequired')}
                        </span>
                      )}
                    </span>
                  </SelectItem>
                ))}
              </SelectGroup>
            ))}
          </SelectContent>
        </Select>
      </div>
      <Button
        aria-label={isBusy ? t('composer.stop') : t('composer.send')}
        className="size-11 shrink-0 rounded-full transition-transform active:scale-95 sm:pointer-fine:size-9"
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
