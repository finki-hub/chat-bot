'use client';

import { ArrowLeft, GraduationCap, RefreshCw, Users } from 'lucide-react';
import Link from 'next/link';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import type {
  CommitteeAlternative,
  CommitteeResponse,
  Professor,
} from '@/lib/committee-types';

import { AlternativesSection } from '@/components/committee/committee-alternatives';
import { CopyButton } from '@/components/committee/committee-copy-button';
import { EvidenceBlock } from '@/components/committee/committee-evidence';
import { CommitteeForm } from '@/components/committee/committee-form';
import { NextSteps } from '@/components/committee/committee-next-steps';
import { PersonCard } from '@/components/committee/committee-person-card';
import {
  type CommitteeRefinements,
  MAX_PINNED_MEMBERS,
  PersonActions,
  RefinementBar,
} from '@/components/committee/committee-refinements';
import { Button } from '@/components/ui/button';
import { Spinner } from '@/components/ui/spinner';
import {
  clearCommitteeDraft,
  type CommitteeFormState as FormState,
  readCommitteeDraft,
  writeCommitteeDraft,
} from '@/lib/committee-draft';
import { t } from '@/lib/i18n';
import { cn } from '@/lib/utils';

const MEMBER_OPTIONS = 3;
const MENTOR_SHORTLIST = 5;

const EMPTY_FORM: FormState = {
  abstract: '',
  hasMentor: false,
  keywords: [],
  mentor: '',
  researchArea: '',
  studyProgram: '',
  title: '',
};

const EMPTY_REFINEMENTS: CommitteeRefinements = { excluded: [], pinned: [] };

const confidenceLevel = (reasons: string[]): 'high' | 'low' | 'medium' => {
  if (
    reasons.includes('weak_topic_match') ||
    reasons.includes('incomplete_committee') ||
    reasons.includes('review_manually')
  ) {
    return 'low';
  }

  return reasons.includes('strong_topic_match') &&
    reasons.includes('enough_similar_defenses')
    ? 'high'
    : 'medium';
};

const CONFIDENCE_MESSAGES = {
  high: t('committee.confidence.high'),
  low: t('committee.confidence.low'),
  medium: t('committee.confidence.medium'),
};

const ConfidenceNote = ({ reasons }: { reasons: string[] }) => {
  const level = confidenceLevel(reasons);

  return (
    <p
      className={cn(
        'mt-2 rounded-lg px-3 py-2 text-sm',
        level === 'high' && 'bg-primary/10 text-foreground',
        level === 'medium' && 'bg-muted text-foreground',
        level === 'low' && 'bg-destructive/10 text-foreground',
      )}
      data-testid="committee-confidence"
    >
      {CONFIDENCE_MESSAGES[level]}
    </p>
  );
};

const buildMentorMessage = (form: FormState, mentor: string): string => {
  const abstract = form.abstract.trim();

  return [
    `${t('committee.message.greeting')} ${mentor},`,
    `${t('committee.message.body')} „${form.title.trim()}“.`,
    ...(abstract ? [`${t('committee.message.abstract')}: ${abstract}`] : []),
    t('committee.message.closing'),
    t('committee.message.signature'),
  ].join('\n\n');
};

const buildSummary = (form: FormState, response: CommitteeResponse): string => {
  const mentorLabel = response.mentor_is_given
    ? t('committee.result.yourMentor')
    : t('committee.result.mentorLabel');
  const members = response.members.map((member) => member.name).join(', ');

  return [
    form.title.trim(),
    `${mentorLabel}: ${response.mentor.name}`,
    `${t('committee.result.membersLabel')}: ${members}`,
    t('committee.disclaimer'),
  ].join('\n');
};

const mergeKeywords = (existing: string[], draft: string): string[] => {
  const keywords = [...existing];
  for (const part of draft.split(',')) {
    const keyword = part.trim().toLowerCase();
    if (keyword && !keywords.includes(keyword)) keywords.push(keyword);
  }
  return keywords;
};

const buildRequestBody = (
  form: FormState,
  keywords: string[],
  refinements: CommitteeRefinements,
) => {
  const hasGivenMentor = form.hasMentor && form.mentor !== '';
  const body: Record<string, unknown> = {
    alternatives: hasGivenMentor ? MEMBER_OPTIONS : MENTOR_SHORTLIST,
    title: form.title.trim(),
  };

  const includePriorKey = 'include_prior';
  body[includePriorKey] = false;

  if (!hasGivenMentor) {
    const distinctMentorsKey = 'distinct_mentors';
    const mentorTopKKey = 'mentor_topk';
    body[distinctMentorsKey] = true;
    body[mentorTopKKey] = MENTOR_SHORTLIST;
  }
  const givenMentor = hasGivenMentor ? form.mentor : null;
  const pinned = refinements.pinned.filter((name) => name !== givenMentor);
  const excluded = refinements.excluded.filter((name) => name !== givenMentor);
  if (pinned.length > 0) {
    const includeKey = 'include_professors';
    const includeAsMembersKey = 'include_as_members';
    body[includeKey] = pinned;
    body[includeAsMembersKey] = true;
  }
  if (excluded.length > 0) {
    const excludeKey = 'exclude_professors';
    body[excludeKey] = excluded;
  }

  if (form.abstract.trim()) body['abstract'] = form.abstract.trim();
  if (keywords.length > 0) body['keywords'] = keywords;
  if (form.researchArea.trim())
    body['research_area'] = form.researchArea.trim();
  if (form.studyProgram.trim())
    body['study_program'] = form.studyProgram.trim();
  if (form.hasMentor && form.mentor) body['mentor'] = form.mentor;

  return body;
};

const isMentorError = (detail: string): boolean =>
  detail.toLowerCase().includes('mentor must');

type SubmissionState =
  | { form: FormState; kind: 'success'; response: CommitteeResponse }
  | { kind: 'error'; message: string; status?: number }
  | { kind: 'idle' }
  | { kind: 'loading' };

const errorState = (message: string, status?: number): SubmissionState => ({
  kind: 'error',
  message,
  status,
});

const parseValidationError = async (
  response: Response,
): Promise<{ mentorError: boolean; message: string }> => {
  const data = (await response.json()) as { detail?: unknown };
  const detail = typeof data.detail === 'string' ? data.detail : '';

  if (detail !== '' && isMentorError(detail)) {
    return { mentorError: true, message: t('committee.error.unknownMentor') };
  }

  return {
    mentorError: false,
    message: detail || t('committee.error.invalidRequest'),
  };
};

const isDegenerateResponse = (response: CommitteeResponse): boolean =>
  response.members.length !== 2 ||
  (!response.mentor_is_given && response.mentor.name === '');

export const CommitteePlanner = () => {
  const resultRef = useRef<HTMLDivElement>(null);
  const activeRequestRef = useRef<AbortController | null>(null);

  const [professors, setProfessors] = useState<Professor[]>([]);
  const [professorsLoading, setProfessorsLoading] = useState(true);
  const [professorsError, setProfessorsError] = useState(false);

  const [form, setForm] = useState<FormState>(EMPTY_FORM);
  const [refinements, setRefinements] =
    useState<CommitteeRefinements>(EMPTY_REFINEMENTS);
  const [draftReady, setDraftReady] = useState(false);

  const [keywordDraft, setKeywordDraft] = useState('');
  const [submission, setSubmission] = useState<SubmissionState>({
    kind: 'idle',
  });
  const [titleError, setTitleError] = useState(false);
  const [mentorError, setMentorError] = useState(false);

  useEffect(() => {
    const draft = readCommitteeDraft();
    if (draft !== null) {
      setForm(draft.form);
      setKeywordDraft(draft.keywordDraft);
    }
    setDraftReady(true);
  }, []);

  useEffect(() => {
    if (draftReady) writeCommitteeDraft({ form, keywordDraft });
  }, [draftReady, form, keywordDraft]);

  useEffect(
    () => () => {
      activeRequestRef.current?.abort();
      activeRequestRef.current = null;
    },
    [],
  );

  const professorCollator = useMemo(
    () => new Intl.Collator('mk', { sensitivity: 'base' }),
    [],
  );

  const loadProfessors = useCallback(async () => {
    setProfessorsLoading(true);
    setProfessorsError(false);

    try {
      const response = await fetch('/api/professors');
      if (!response.ok) {
        throw new Error('directory unavailable');
      }
      const data = (await response.json()) as Professor[];
      const sorted = data.toSorted((a, b) =>
        professorCollator.compare(a.name, b.name),
      );
      setProfessors(sorted);
    } catch {
      setProfessorsError(true);
    } finally {
      setProfessorsLoading(false);
    }
  }, [professorCollator]);

  useEffect(() => {
    void loadProfessors();
  }, [loadProfessors]);

  const validate = (candidate: FormState): boolean => {
    const isTitleValid = candidate.title.trim().length > 0;
    const isMentorValid = !candidate.hasMentor || candidate.mentor !== '';
    setTitleError(!isTitleValid);
    setMentorError(!isMentorValid);
    if (!isTitleValid) {
      document.querySelector<HTMLElement>('#committee-title')?.focus();
    } else if (!isMentorValid) {
      document.querySelector<HTMLElement>('#committee-mentor-mode')?.focus();
    }
    return isTitleValid && isMentorValid;
  };

  const requestCommittee = async (
    nextForm: FormState,
    nextRefinements: CommitteeRefinements,
  ) => {
    const keywords = mergeKeywords(nextForm.keywords, keywordDraft);

    activeRequestRef.current?.abort();
    const controller = new AbortController();
    activeRequestRef.current = controller;
    const isStale = () => activeRequestRef.current !== controller;
    setSubmission({ kind: 'loading' });

    try {
      const response = await fetch('/api/committee', {
        body: JSON.stringify(
          buildRequestBody(nextForm, keywords, nextRefinements),
        ),
        headers: { 'content-type': 'application/json' },
        method: 'POST',
        signal: controller.signal,
      });

      if (isStale()) return;

      if (response.status === 400) {
        setSubmission(errorState(t('committee.error.invalidRequest'), 400));
        return;
      }

      if (response.status === 404) {
        setSubmission(errorState(t('committee.error.notFound'), 404));
        return;
      }

      if (response.status === 429) {
        setSubmission(errorState(t('committee.error.tooManyRequests'), 429));
        return;
      }

      if (response.status === 422) {
        const validation = await parseValidationError(response);
        if (isStale()) return;
        setSubmission(errorState(validation.message, 422));
        if (validation.mentorError) setMentorError(true);
        return;
      }

      if (!response.ok) {
        setSubmission(errorState(t('committee.error.networkError')));
        return;
      }

      const responseData = (await response.json()) as CommitteeResponse;
      if (isStale()) return;

      if (isDegenerateResponse(responseData)) {
        setSubmission(errorState(t('committee.error.emptyCommittee')));
        return;
      }

      setSubmission({
        form: nextForm,
        kind: 'success',
        response: responseData,
      });
    } catch {
      if (isStale()) return;
      setSubmission(errorState(t('committee.error.networkError')));
    }
  };

  const handleSubmit = async () => {
    if (!validate(form)) return;
    await requestCommittee(form, refinements);
  };

  const refine = (
    nextRefinements: CommitteeRefinements,
    nextForm: FormState = form,
  ) => {
    if (!validate(nextForm)) return;
    setForm(nextForm);
    setRefinements(nextRefinements);
    void requestCommittee(nextForm, nextRefinements);
  };

  const handleChooseMentor = (name: string) => {
    refine(
      {
        excluded: refinements.excluded.filter((item) => item !== name),
        pinned: refinements.pinned.filter((item) => item !== name),
      },
      { ...form, hasMentor: true, mentor: name },
    );
  };

  const handleExclude = (name: string) => {
    refine({
      excluded: [...refinements.excluded, name],
      pinned: refinements.pinned.filter((item) => item !== name),
    });
  };

  const handleTogglePin = (name: string) => {
    const isPinned = refinements.pinned.includes(name);
    refine({
      ...refinements,
      pinned: isPinned
        ? refinements.pinned.filter((item) => item !== name)
        : [...refinements.pinned, name],
    });
  };

  const handleRemoveRefinement = (name: string) => {
    refine({
      excluded: refinements.excluded.filter((item) => item !== name),
      pinned: refinements.pinned.filter((item) => item !== name),
    });
  };

  const handleChooseAlternative = (alternative: CommitteeAlternative) => {
    if (submission.kind !== 'success') return;

    if (submission.response.mentor_is_given) {
      refine({
        excluded: refinements.excluded.filter(
          (item) => !alternative.members.includes(item),
        ),
        pinned: alternative.members,
      });
    } else if (alternative.mentor !== null) {
      handleChooseMentor(alternative.mentor);
    }
  };

  useEffect(() => {
    if (submission.kind === 'success' || submission.kind === 'error') {
      resultRef.current?.focus();
    }
  }, [submission]);

  const handleRestart = () => {
    activeRequestRef.current?.abort();
    activeRequestRef.current = null;
    setSubmission({ kind: 'idle' });
    setForm(EMPTY_FORM);
    setRefinements(EMPTY_REFINEMENTS);
    setKeywordDraft('');
    setTitleError(false);
    setMentorError(false);
    clearCommitteeDraft();
    document.querySelector<HTMLElement>('#committee-title')?.focus();
  };

  const filteredAlternatives = useMemo<CommitteeAlternative[]>(() => {
    if (submission.kind !== 'success') return [];

    const selected = submission.response;
    const selectedMembers = new Set(
      selected.members.map((member) => member.name),
    );

    return selected.alternatives.filter((alternative) => {
      if (alternative.mentor === null || alternative.members.length !== 2) {
        return false;
      }

      const sameMentor = alternative.mentor === selected.mentor.name;
      const sameMembers =
        alternative.members.length === selectedMembers.size &&
        alternative.members.every((member) => selectedMembers.has(member));

      return !(sameMentor && sameMembers);
    });
  }, [submission]);

  return (
    <div className="mx-auto flex min-h-dvh max-w-3xl flex-col px-4 py-6 sm:px-6 lg:py-10">
      <div className="mb-6 flex items-center justify-between gap-4">
        <Link
          className="inline-flex items-center gap-1.5 text-sm font-medium text-muted-foreground transition-colors hover:text-foreground"
          href="/"
        >
          <ArrowLeft className="size-4" />
          {t('committee.backToChat')}
        </Link>
        <a
          className="inline-flex items-center gap-1.5 text-sm font-medium text-muted-foreground transition-colors hover:text-foreground"
          href="https://diplomski.finki.ukim.mk/"
          rel="noopener noreferrer"
          target="_blank"
        >
          <GraduationCap className="size-4" />
          {t('committee.diplomasLink')}
        </a>
      </div>

      <header className="mb-6 flex items-start gap-3 sm:mb-8 sm:gap-4">
        <img
          alt=""
          aria-hidden="true"
          className="mt-1 size-8 shrink-0 object-contain sm:size-10"
          src="/logo.png"
        />
        <div className="space-y-1">
          <h1 className="text-balance text-2xl font-bold tracking-tight sm:text-4xl">
            {t('committee.title')}
          </h1>
          <p className="text-pretty text-sm text-muted-foreground sm:text-lg">
            {t('committee.pageTitle')}
          </p>
        </div>
      </header>

      <CommitteeForm
        form={form}
        keywordDraft={keywordDraft}
        mentorError={mentorError}
        onLoadProfessors={loadProfessors}
        onSubmit={handleSubmit}
        professors={professors}
        professorsError={professorsError}
        professorsLoading={professorsLoading}
        setForm={setForm}
        setKeywordDraft={setKeywordDraft}
        setMentorError={setMentorError}
        setTitleError={setTitleError}
        submitting={submission.kind === 'loading'}
        titleError={titleError}
      />

      <div
        aria-live="polite"
        className="mt-8 scroll-mt-4 outline-none"
        data-testid="committee-result-region"
        ref={resultRef}
        tabIndex={-1}
      >
        {submission.kind === 'idle' ? null : (
          <RefinementBar
            onRemove={handleRemoveRefinement}
            refinements={refinements}
          />
        )}

        {submission.kind === 'loading' ? (
          <div
            className="flex items-center gap-3 rounded-xl border border-border bg-card p-4 text-sm text-muted-foreground shadow-sm"
            data-testid="committee-loading"
          >
            <Spinner className="size-5" />
            {t('committee.submitting')}
          </div>
        ) : null}

        {submission.kind === 'error' ? (
          <div
            className="rounded-xl border border-destructive/30 bg-destructive/10 p-4 text-sm text-destructive"
            data-testid="committee-error"
            role="alert"
          >
            {submission.message}
          </div>
        ) : null}

        {submission.kind === 'success' ? (
          <div
            className="space-y-6"
            data-testid="committee-result"
          >
            <div className="rounded-xl border border-border bg-card p-4 shadow-sm">
              <div className="mb-4 flex items-start gap-3">
                <Users className="mt-0.5 size-5 shrink-0 text-primary" />
                <div>
                  <h2 className="text-lg font-semibold text-foreground">
                    {t('committee.result.title')}
                  </h2>
                  <p className="text-sm text-muted-foreground">
                    {t('committee.disclaimer')}
                  </p>
                  <ConfidenceNote
                    reasons={submission.response.confidence_reasons}
                  />
                </div>
              </div>

              <div className="space-y-3">
                <PersonCard
                  actions={
                    submission.response.mentor_is_given ? null : (
                      <PersonActions
                        canPin={false}
                        isPinned={false}
                        name={submission.response.mentor.name}
                        onChooseMentor={handleChooseMentor}
                        onExclude={handleExclude}
                      />
                    )
                  }
                  featured
                  label={
                    submission.response.mentor_is_given
                      ? t('committee.result.yourMentor')
                      : t('committee.result.mentorLabel')
                  }
                  person={submission.response.mentor}
                  professors={professors}
                />
                <div className="grid gap-3 sm:grid-cols-2">
                  {submission.response.members.map((member) => (
                    <PersonCard
                      actions={
                        <PersonActions
                          canPin={
                            refinements.pinned.length < MAX_PINNED_MEMBERS
                          }
                          isPinned={refinements.pinned.includes(member.name)}
                          name={member.name}
                          onExclude={handleExclude}
                          onTogglePin={handleTogglePin}
                        />
                      }
                      key={member.name}
                      label={t('committee.result.memberLabel')}
                      person={member}
                      professors={professors}
                    />
                  ))}
                </div>
              </div>

              <div className="mt-4 flex flex-col flex-wrap gap-2 sm:flex-row">
                <CopyButton
                  getText={() =>
                    buildMentorMessage(
                      submission.form,
                      submission.response.mentor.name,
                    )
                  }
                  label={t('committee.copyMessage')}
                />
                <CopyButton
                  getText={() =>
                    buildSummary(submission.form, submission.response)
                  }
                  label={t('committee.copySummary')}
                />
                <Button
                  className="pointer-coarse:min-h-11"
                  onClick={handleRestart}
                  type="button"
                  variant="outline"
                >
                  <RefreshCw className="size-4" />
                  {t('committee.startOver')}
                </Button>
              </div>
            </div>

            <AlternativesSection
              alternatives={filteredAlternatives}
              mentorIsGiven={submission.response.mentor_is_given}
              onChoose={handleChooseAlternative}
              professors={professors}
              selectedMentor={submission.response.mentor.name}
            />

            <NextSteps />

            <EvidenceBlock
              diplomas={submission.response.evidence.similar_diplomas}
            />
          </div>
        ) : null}
      </div>
    </div>
  );
};
