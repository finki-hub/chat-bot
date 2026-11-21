import type { Professor } from '@/lib/committee-types';

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Spinner } from '@/components/ui/spinner';
import { t } from '@/lib/i18n';

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
      <Select
        onValueChange={onSelect}
        value={mentor}
      >
        <SelectTrigger
          aria-describedby={mentorError ? `${selectId}-error` : undefined}
          aria-invalid={mentorError}
          className="w-full"
          data-testid="committee-mentor-select"
          id={selectId}
        >
          <SelectValue placeholder={t('committee.mentorPicker')} />
        </SelectTrigger>
        <SelectContent className="max-h-72">
          {eligible.map((professor) => (
            <SelectItem
              key={professor.name}
              value={professor.name}
            >
              {professor.name}
              {professor.position ? ` · ${professor.position}` : null}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {mentorError ? (
        <p
          className="text-xs text-destructive"
          id={`${selectId}-error`}
        >
          {t('committee.error.unknownMentor')}
        </p>
      ) : null}
    </>
  );
};
