'use client';

import { ArrowLeft, Copy, GraduationCap, RefreshCw, Users } from 'lucide-react';
import Link from 'next/link';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import type {
  CommitteeAlternative,
  CommitteeResponse,
  Professor,
} from '@/lib/committee-types';

import { AlternativesSection } from '@/components/committee/committee-alternatives';
import { EvidenceBlock } from '@/components/committee/committee-evidence';
import { CommitteeForm } from '@/components/committee/committee-form';
import { PersonCard } from '@/components/committee/committee-person-card';
import { Button } from '@/components/ui/button';
import { Spinner } from '@/components/ui/spinner';
import { t } from '@/lib/i18n';

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

const CONFIDENCE_REASON_LABELS: Record<string, string> = {
  'enough similar defenses': t('committee.confidence.enoughSimilarDefenses'),
  'incomplete committee': t('committee.confidence.incompleteCommittee'),
  'limited similar defenses': t('committee.confidence.limitedSimilarDefenses'),
  'moderate topic match': t('committee.confidence.moderateTopicMatch'),
  'review manually': t('committee.confidence.reviewManually'),
  'strong topic match': t('committee.confidence.strongTopicMatch'),
  'weak topic match': t('committee.confidence.weakTopicMatch'),
};

const mapConfidenceReason = (reason: string): null | string => {
  const normalized = reason.toLowerCase().trim().replaceAll('_', ' ');
  return CONFIDENCE_REASON_LABELS[normalized] ?? null;
};

const ConfidenceSection = ({ reasons }: { reasons: string[] }) => {
  const labels = reasons.map(mapConfidenceReason).filter(Boolean);
  if (labels.length === 0) return null;

  return (
    <div className="flex flex-wrap gap-2">
      {labels.map((label) => (
        <span
          className="rounded-full bg-muted px-2.5 py-1 text-xs font-medium text-muted-foreground"
          key={label}
        >
          {label}
        </span>
      ))}
    </div>
  );
};

const buildRequestBody = (form: FormState, keywords: string[]) => {
  const body: Record<string, unknown> = {
    alternatives: form.alternatives,
    title: form.title.trim(),
  };

  const includePriorKey = 'include_prior';
  body[includePriorKey] = false;

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
  detail.toLowerCase().includes('mentor') ||
  detail.toLowerCase().includes('active staff');

type SubmissionState =
  | { kind: 'error'; message: string; status?: number }
  | { kind: 'idle' }
  | { kind: 'loading' }
  | { kind: 'success'; response: CommitteeResponse };

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

  const [professors, setProfessors] = useState<Professor[]>([]);
  const [professorsLoading, setProfessorsLoading] = useState(true);
  const [professorsError, setProfessorsError] = useState(false);

  const [form, setForm] = useState<FormState>({
    abstract: '',
    alternatives: 3,
    hasMentor: false,
    keywords: [],
    mentor: '',
    researchArea: '',
    studyProgram: '',
    title: '',
  });

  const [keywordDraft, setKeywordDraft] = useState('');
  const [submission, setSubmission] = useState<SubmissionState>({
    kind: 'idle',
  });
  const [titleError, setTitleError] = useState(false);
  const [mentorError, setMentorError] = useState(false);

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

  const validate = (): boolean => {
    const isTitleValid = form.title.trim().length > 0;
    const isMentorValid = !form.hasMentor || form.mentor !== '';
    setTitleError(!isTitleValid);
    setMentorError(!isMentorValid);
    return isTitleValid && isMentorValid;
  };

  const handleSubmit = async () => {
    if (!validate()) return;

    const pendingKeywords = keywordDraft
      .split(',')
      .map((part) => part.trim().toLowerCase())
      .filter((part) => part.length > 0);
    const keywords = [...form.keywords];
    for (const keyword of pendingKeywords) {
      if (!keywords.includes(keyword)) keywords.push(keyword);
    }

    setSubmission({ kind: 'loading' });

    try {
      const response = await fetch('/api/committee', {
        body: JSON.stringify(buildRequestBody(form, keywords)),
        headers: { 'content-type': 'application/json' },
        method: 'POST',
      });

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
        setSubmission(errorState(validation.message, 422));
        if (validation.mentorError) setMentorError(true);
        return;
      }

      if (!response.ok) {
        setSubmission(errorState(t('committee.error.networkError')));
        return;
      }

      const responseData = (await response.json()) as CommitteeResponse;

      if (isDegenerateResponse(responseData)) {
        setSubmission(errorState(t('committee.error.emptyCommittee')));
        return;
      }

      setSubmission({ kind: 'success', response: responseData });
    } catch {
      setSubmission(errorState(t('committee.error.networkError')));
    }
  };

  useEffect(() => {
    if (submission.kind === 'success' || submission.kind === 'error') {
      resultRef.current?.focus();
    }
  }, [submission]);

  const handleCopySummary = async () => {
    if (submission.kind !== 'success') return;

    const { response } = submission;
    const mentorLine = response.mentor_is_given
      ? `${t('committee.result.yourMentor')}: ${response.mentor.name}`
      : `${t('committee.result.mentorLabel')}: ${response.mentor.name}`;
    const membersLine = `${t('committee.result.membersLabel')}: ${response.members
      .map((member) => member.name)
      .join(', ')}`;
    const disclaimer = t('committee.disclaimer');

    try {
      await navigator.clipboard.writeText(
        [form.title.trim(), mentorLine, membersLine, disclaimer].join('\n'),
      );
    } catch {
      // Clipboard access can be denied by the browser; the summary stays visible.
    }
  };

  const handleRestart = () => {
    setSubmission({ kind: 'idle' });
    setForm((previous) => ({
      ...previous,
      abstract: '',
      keywords: [],
      researchArea: '',
      studyProgram: '',
      title: '',
    }));
    setKeywordDraft('');
    setTitleError(false);
    setMentorError(false);
  };

  const filteredAlternatives = useMemo<CommitteeAlternative[]>(
    () =>
      submission.kind === 'success'
        ? submission.response.alternatives.filter(
            (alternative) =>
              alternative.mentor !== null && alternative.members.length === 2,
          )
        : [],
    [submission],
  );

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

      <header className="mb-8 space-y-2">
        <div className="flex items-center gap-3">
          <img
            alt=""
            aria-hidden="true"
            className="h-10 w-10 object-contain"
            src="/logo.png"
          />
          <h1 className="text-balance text-3xl font-bold tracking-tight sm:text-4xl">
            {t('committee.title')}
          </h1>
        </div>
        <p className="text-pretty text-base text-muted-foreground sm:text-lg">
          {t('committee.pageTitle')}
        </p>
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
        className="mt-8"
        data-testid="committee-result-region"
        ref={resultRef}
        tabIndex={-1}
      >
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
                  <div className="mt-2">
                    <ConfidenceSection
                      reasons={submission.response.confidence_reasons}
                    />
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                <PersonCard
                  evidence={submission.response.evidence}
                  label={
                    submission.response.mentor_is_given
                      ? t('committee.result.yourMentor')
                      : t('committee.result.mentorLabel')
                  }
                  person={submission.response.mentor}
                  professors={professors}
                />
                <div>
                  <h3 className="mb-2 text-sm font-medium text-muted-foreground">
                    {t('committee.result.membersLabel')}
                  </h3>
                  <div className="grid gap-3 sm:grid-cols-2">
                    {submission.response.members.map((member) => (
                      <PersonCard
                        evidence={submission.response.evidence}
                        key={member.name}
                        label={t('committee.result.membersLabel')}
                        person={member}
                        professors={professors}
                      />
                    ))}
                  </div>
                </div>
              </div>
            </div>

            <AlternativesSection
              alternatives={filteredAlternatives}
              professors={professors}
            />

            <EvidenceBlock
              diplomas={submission.response.evidence.similar_diplomas}
            />

            <div className="flex flex-col gap-3 sm:flex-row">
              <Button
                className="pointer-coarse:min-h-11"
                onClick={() => {
                  void handleCopySummary();
                }}
                type="button"
                variant="outline"
              >
                <Copy className="size-4" />
                {t('committee.copySummary')}
              </Button>
              <Button
                className="pointer-coarse:min-h-11"
                onClick={handleRestart}
                type="button"
                variant="outline"
              >
                <RefreshCw className="size-4" />
                {t('committee.startOver')}
              </Button>
              <Button
                asChild
                className="pointer-coarse:min-h-11"
                variant="outline"
              >
                <a
                  href="https://diplomski.finki.ukim.mk/"
                  rel="noopener noreferrer"
                  target="_blank"
                >
                  <GraduationCap className="size-4" />
                  {t('committee.diplomasLink')}
                </a>
              </Button>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
};
