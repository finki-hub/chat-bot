import { Check, ChevronsUpDown } from 'lucide-react';
import { Popover } from 'radix-ui';
import { useState } from 'react';

import type { Professor } from '@/lib/committee-types';

import { Button } from '@/components/ui/button';
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from '@/components/ui/command';
import { Spinner } from '@/components/ui/spinner';
import { t } from '@/lib/i18n';
import { cn } from '@/lib/utils';

type MentorPickerProps = {
  mentor: string;
  mentorError: boolean;
  onLoadProfessors: () => void;
  onSelect: (value: string) => void;
  professors: Professor[];
  professorsError: boolean;
  professorsLoading: boolean;
  selectId: string;
};

export const MentorPicker = ({
  mentor,
  mentorError,
  onLoadProfessors,
  onSelect,
  professors,
  professorsError,
  professorsLoading,
  selectId,
}: MentorPickerProps) => {
  const [open, setOpen] = useState(false);

  if (professorsLoading) {
    return (
      <div className="flex h-9 items-center gap-2 text-sm text-muted-foreground">
        <Spinner />
        {t('committee.loadingStaff')}
      </div>
    );
  }

  if (professorsError) {
    return (
      <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">
        {t('committee.error.directoryUnavailable')}
        <button
          className="ml-2 underline"
          onClick={onLoadProfessors}
          type="button"
        >
          {t('error.retry')}
        </button>
      </div>
    );
  }

  const eligible = professors.filter((professor) => professor.mentor_eligible);
  if (eligible.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        {t('committee.error.noEligibleMentors')}
      </p>
    );
  }

  return (
    <>
      <Popover.Root
        onOpenChange={setOpen}
        open={open}
      >
        <Popover.Trigger asChild>
          <Button
            aria-describedby={mentorError ? `${selectId}-error` : undefined}
            aria-invalid={mentorError}
            className="w-full justify-between font-normal pointer-coarse:min-h-11"
            data-testid="committee-mentor-select"
            id={selectId}
            type="button"
            variant="outline"
          >
            <span
              className={cn('truncate', !mentor && 'text-muted-foreground')}
            >
              {mentor || t('committee.mentorPicker')}
            </span>
            <ChevronsUpDown className="size-4 shrink-0 opacity-50" />
          </Button>
        </Popover.Trigger>
        <Popover.Portal>
          <Popover.Content
            align="start"
            className="z-50 w-(--radix-popover-trigger-width) overflow-hidden rounded-md border border-border bg-popover text-popover-foreground shadow-md"
            sideOffset={4}
          >
            <Command>
              <CommandInput placeholder={t('committee.mentorSearch')} />
              <CommandList className="max-h-72">
                <CommandEmpty>{t('committee.mentorSearchEmpty')}</CommandEmpty>
                <CommandGroup>
                  {eligible.map((professor) => (
                    <CommandItem
                      className="group pointer-coarse:min-h-11"
                      key={professor.name}
                      keywords={professor.position ? [professor.position] : []}
                      onSelect={() => {
                        onSelect(professor.name);
                        setOpen(false);
                      }}
                      value={professor.name}
                    >
                      <span className="min-w-0 flex-1">
                        <span className="block truncate">{professor.name}</span>
                        {professor.position ? (
                          <span className="block truncate text-xs text-muted-foreground group-data-[selected=true]:text-accent-foreground">
                            {professor.position}
                          </span>
                        ) : null}
                      </span>
                      <Check
                        className={cn(
                          'size-4',
                          mentor === professor.name
                            ? 'opacity-100'
                            : 'opacity-0',
                        )}
                      />
                    </CommandItem>
                  ))}
                </CommandGroup>
              </CommandList>
            </Command>
          </Popover.Content>
        </Popover.Portal>
      </Popover.Root>
      {mentorError ? (
        <p
          className="text-xs text-destructive"
          id={`${selectId}-error`}
        >
          {t(
            mentor
              ? 'committee.error.unknownMentor'
              : 'committee.error.mentorRequired',
          )}
        </p>
      ) : null}
    </>
  );
};
