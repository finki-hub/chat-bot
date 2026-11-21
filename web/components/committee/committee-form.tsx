import type { ChangeEvent, KeyboardEvent } from 'react';

import { X } from 'lucide-react';

import type { Professor } from '@/lib/committee-types';

import { MentorPicker } from '@/components/committee/committee-mentor-picker';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Spinner } from '@/components/ui/spinner';
import { Textarea } from '@/components/ui/textarea';
import { t } from '@/lib/i18n';
import { cn } from '@/lib/utils';

type CommitteeFormProps = {
  form: FormState;
  keywordDraft: string;
  mentorError: boolean;
  onLoadProfessors: () => void;
  onSubmit: () => void;
  professors: Professor[];
  professorsError: boolean;
  professorsLoading: boolean;
  setForm: (value: ((previous: FormState) => FormState) | FormState) => void;
  setKeywordDraft: (value: string) => void;
  setMentorError: (value: boolean) => void;
  setTitleError: (value: boolean) => void;
  submitting: boolean;
  titleError: boolean;
};

type FormState = {
  abstract: string;
  alternatives: 1 | 2 | 3;
  hasMentor: boolean;
  keywords: string[];
  mentor: string;
  researchArea: string;
  studyProgram: string;
  title: string;
};

const ALTERNATIVE_OPTIONS = [1, 2, 3] as const;
const keywordInputId = 'committee-keywords';

export const CommitteeForm = ({
  form,
  keywordDraft,
  mentorError,
  onLoadProfessors,
  onSubmit,
  professors,
  professorsError,
  professorsLoading,
  setForm,
  setKeywordDraft,
  setMentorError,
  setTitleError,
  submitting,
  titleError,
}: CommitteeFormProps) => {
  const titleId = 'committee-title';
  const abstractId = 'committee-abstract';
  const researchAreaId = 'committee-research-area';
  const studyProgramId = 'committee-study-program';
  const mentorFieldsetId = 'committee-mentor-mode';
  const alternativesId = 'committee-alternatives';
  const titleDescription = titleError
    ? `${titleId}-help ${titleId}-error`
    : `${titleId}-help`;

  const addKeyword = () => {
    const normalized = keywordDraft
      .split(',')
      .map((part) => part.trim().toLowerCase())
      .filter((part) => part.length > 0);
    if (normalized.length === 0) return;

    setForm((previous) => {
      const keywords = [...previous.keywords];
      for (const keyword of normalized) {
        if (!keywords.includes(keyword)) keywords.push(keyword);
      }
      return { ...previous, keywords };
    });
    setKeywordDraft('');
  };

  const removeKeyword = (keyword: string) => {
    setForm((previous) => ({
      ...previous,
      keywords: previous.keywords.filter((item) => item !== keyword),
    }));
  };

  const handleKeywordKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter' || event.key === ',') {
      event.preventDefault();
      addKeyword();
    } else if (keywordDraft === '' && event.key === 'Backspace') {
      setForm((previous) => ({
        ...previous,
        keywords: previous.keywords.slice(0, -1),
      }));
    }
  };

  return (
    <form
      className="space-y-5"
      data-testid="committee-form"
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit();
      }}
    >
      <div className="space-y-1.5">
        <label
          className="text-sm font-medium text-foreground"
          htmlFor={titleId}
        >
          {t('committee.titleField')}
        </label>
        <Input
          aria-describedby={titleDescription}
          aria-invalid={titleError}
          id={titleId}
          onChange={(event: ChangeEvent<HTMLInputElement>) => {
            setForm((previous) => ({ ...previous, title: event.target.value }));
            if (titleError) setTitleError(false);
          }}
          placeholder={t('committee.titlePlaceholder')}
          value={form.title}
        />
        <p
          className="text-xs text-muted-foreground"
          id={`${titleId}-help`}
        >
          {t('committee.required')}
        </p>
        {titleError ? (
          <p
            className="text-xs text-destructive"
            id={`${titleId}-error`}
            role="alert"
          >
            {t('committee.error.titleRequired')}
          </p>
        ) : null}
      </div>

      <div className="space-y-1.5">
        <label
          className="text-sm font-medium text-foreground"
          htmlFor={abstractId}
        >
          {t('committee.abstract')}
        </label>
        <Textarea
          aria-describedby={`${abstractId}-help`}
          id={abstractId}
          onChange={(event: ChangeEvent<HTMLTextAreaElement>) => {
            setForm((previous) => ({
              ...previous,
              abstract: event.target.value,
            }));
          }}
          placeholder={t('committee.abstractPlaceholder')}
          rows={4}
          value={form.abstract}
        />
        <p
          className="text-xs text-muted-foreground"
          id={`${abstractId}-help`}
        >
          {t('committee.abstractHelp')}
        </p>
      </div>

      <div className="space-y-1.5">
        <label
          className="text-sm font-medium text-foreground"
          htmlFor={keywordInputId}
        >
          {t('committee.keywords')}
        </label>
        <div className="flex min-h-[2.25rem] flex-wrap items-center gap-1.5 rounded-md border border-input bg-transparent px-2 py-1.5 shadow-xs focus-within:border-ring focus-within:ring-[3px] focus-within:ring-ring/50">
          {form.keywords.map((keyword) => (
            <span
              className="inline-flex items-center gap-1 rounded-md bg-secondary px-2 py-0.5 text-xs font-medium text-secondary-foreground"
              key={keyword}
            >
              {keyword}
              <button
                aria-label={`${t('committee.removeKeyword')} ${keyword}`}
                className="inline-flex size-4 items-center justify-center rounded-sm hover:bg-secondary-foreground/10"
                onClick={() => {
                  removeKeyword(keyword);
                }}
                type="button"
              >
                <X className="size-3" />
              </button>
            </span>
          ))}
          <input
            aria-describedby={`${keywordInputId}-help`}
            aria-label={t('committee.keywords')}
            className="min-w-[8rem] flex-1 bg-transparent py-1 text-sm outline-none placeholder:text-muted-foreground"
            data-testid="committee-keywords-input"
            id={keywordInputId}
            onBlur={() => {
              addKeyword();
            }}
            onChange={(event: ChangeEvent<HTMLInputElement>) => {
              setKeywordDraft(event.target.value);
            }}
            onKeyDown={handleKeywordKeyDown}
            placeholder={
              form.keywords.length === 0
                ? t('committee.keywordsPlaceholder')
                : ''
            }
            value={keywordDraft}
          />
        </div>
        <p
          className="text-xs text-muted-foreground"
          id={`${keywordInputId}-help`}
        >
          {t('committee.keywordsHelp')}
        </p>
      </div>

      <div className="grid gap-5 sm:grid-cols-2">
        <div className="space-y-1.5">
          <label
            className="text-sm font-medium text-foreground"
            htmlFor={researchAreaId}
          >
            {t('committee.researchArea')}
          </label>
          <Input
            id={researchAreaId}
            onChange={(event: ChangeEvent<HTMLInputElement>) => {
              setForm((previous) => ({
                ...previous,
                researchArea: event.target.value,
              }));
            }}
            value={form.researchArea}
          />
        </div>

        <div className="space-y-1.5">
          <label
            className="text-sm font-medium text-foreground"
            htmlFor={studyProgramId}
          >
            {t('committee.studyProgram')}
          </label>
          <Input
            id={studyProgramId}
            onChange={(event: ChangeEvent<HTMLInputElement>) => {
              setForm((previous) => ({
                ...previous,
                studyProgram: event.target.value,
              }));
            }}
            value={form.studyProgram}
          />
        </div>
      </div>

      <fieldset className="space-y-3">
        <legend className="text-sm font-medium text-foreground">
          {t('committee.mentorMode.label')}
        </legend>
        <div className="flex flex-col gap-2 sm:flex-row">
          <label
            className={cn(
              'flex cursor-pointer items-center gap-2 rounded-lg border px-3 py-2.5 text-sm transition-colors',
              form.hasMentor
                ? 'border-border bg-card'
                : 'border-primary/40 bg-primary/5',
            )}
            htmlFor={`${mentorFieldsetId}-none`}
          >
            <input
              aria-label={t('committee.mentorMode.none')}
              checked={!form.hasMentor}
              className="size-4"
              id={`${mentorFieldsetId}-none`}
              onChange={() => {
                setForm((previous) => ({
                  ...previous,
                  hasMentor: false,
                  mentor: '',
                }));
                setMentorError(false);
              }}
              type="radio"
              value="none"
            />
            {t('committee.mentorMode.none')}
          </label>
          <label
            className={cn(
              'flex cursor-pointer items-center gap-2 rounded-lg border px-3 py-2.5 text-sm transition-colors',
              form.hasMentor
                ? 'border-primary/40 bg-primary/5'
                : 'border-border bg-card',
            )}
            htmlFor={`${mentorFieldsetId}-given`}
          >
            <input
              aria-label={t('committee.mentorMode.given')}
              checked={form.hasMentor}
              className="size-4"
              id={`${mentorFieldsetId}-given`}
              onChange={() => {
                setForm((previous) => ({ ...previous, hasMentor: true }));
              }}
              type="radio"
              value="given"
            />
            {t('committee.mentorMode.given')}
          </label>
        </div>
      </fieldset>

      {form.hasMentor ? (
        <div className="space-y-1.5">
          <label
            className="text-sm font-medium text-foreground"
            htmlFor={mentorFieldsetId}
          >
            {t('committee.mentorPicker')}
          </label>
          <MentorPicker
            mentor={form.mentor}
            mentorError={mentorError}
            onLoadProfessors={onLoadProfessors}
            onSelect={(value) => {
              setForm((previous) => ({ ...previous, mentor: value }));
              if (mentorError) setMentorError(false);
            }}
            professors={professors}
            professorsError={professorsError}
            professorsLoading={professorsLoading}
            selectId={mentorFieldsetId}
          />
        </div>
      ) : null}

      <div className="space-y-1.5">
        <label
          className="text-sm font-medium text-foreground"
          htmlFor={alternativesId}
        >
          {t('committee.alternatives')}
        </label>
        <Select
          onValueChange={(value) => {
            setForm((previous) => ({
              ...previous,
              alternatives: Number(value) as 1 | 2 | 3,
            }));
          }}
          value={String(form.alternatives)}
        >
          <SelectTrigger
            className="w-full"
            data-testid="committee-alternatives-select"
            id={alternativesId}
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {ALTERNATIVE_OPTIONS.map((option) => (
              <SelectItem
                key={option}
                value={String(option)}
              >
                {option}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <Button
        aria-busy={submitting}
        className="w-full pointer-coarse:min-h-11 sm:w-auto"
        data-testid="committee-submit"
        disabled={submitting}
        type="submit"
      >
        {submitting ? (
          <>
            <Spinner className="size-4" />
            {t('committee.submitting')}
          </>
        ) : (
          t('committee.submit')
        )}
      </Button>
    </form>
  );
};
