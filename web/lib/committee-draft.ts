export type CommitteeDraft = {
  form: CommitteeFormState;
  keywordDraft: string;
};

export type CommitteeFormState = {
  abstract: string;
  hasMentor: boolean;
  keywords: string[];
  mentor: string;
  researchArea: string;
  studyProgram: string;
  title: string;
};

const DRAFT_KEY = 'committee-planner-draft-v1';

const isStringArray = (value: unknown): value is string[] =>
  Array.isArray(value) && value.every((item) => typeof item === 'string');

const isDraft = (value: unknown): value is CommitteeDraft => {
  if (typeof value !== 'object' || value === null) return false;

  const { form, keywordDraft } = value as Partial<CommitteeDraft>;
  return (
    typeof keywordDraft === 'string' &&
    typeof form === 'object' &&
    typeof form.abstract === 'string' &&
    typeof form.hasMentor === 'boolean' &&
    isStringArray(form.keywords) &&
    typeof form.mentor === 'string' &&
    typeof form.researchArea === 'string' &&
    typeof form.studyProgram === 'string' &&
    typeof form.title === 'string'
  );
};

export const readCommitteeDraft = (): CommitteeDraft | null => {
  try {
    const raw = localStorage.getItem(DRAFT_KEY);
    if (raw === null) return null;

    const parsed: unknown = JSON.parse(raw);
    return isDraft(parsed) ? parsed : null;
  } catch {
    return null;
  }
};

export const clearCommitteeDraft = (): void => {
  try {
    localStorage.removeItem(DRAFT_KEY);
  } catch {
    // Storage can be unavailable in private browsing.
  }
};

export const writeCommitteeDraft = (draft: CommitteeDraft): void => {
  const { form, keywordDraft } = draft;
  const isEmpty =
    !form.title &&
    !form.abstract &&
    form.keywords.length === 0 &&
    !form.researchArea &&
    !form.studyProgram &&
    !form.hasMentor &&
    !keywordDraft;

  if (isEmpty) {
    clearCommitteeDraft();
    return;
  }

  try {
    localStorage.setItem(DRAFT_KEY, JSON.stringify(draft));
  } catch {
    // Saving the draft is best effort.
  }
};
