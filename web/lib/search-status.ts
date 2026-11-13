import {
  FileText,
  type LucideIcon,
  Megaphone,
  MessageCircleQuestion,
  Search,
  Wrench,
} from 'lucide-react';

import { t, type TKey } from '@/lib/i18n';

export const SEARCH_STAGES = [
  'contextualize',
  'retrieve',
  'rerank',
  'context',
  'generate',
] as const;

export type SearchStage = (typeof SEARCH_STAGES)[number];

/* eslint-disable camelcase -- keys are backend tool names, not camelCase identifiers */
const TOOL_ICONS: Record<string, LucideIcon> = {
  faq_search: MessageCircleQuestion,
  search_docs: FileText,
  search_documents: FileText,
  search_knowledge_base: Search,
};

const TOOL_LABELS: Record<string, string> = {
  faq_search: t('searchTool.faq_search'),
  search_docs: t('searchTool.search_docs'),
  search_documents: t('searchTool.search_documents'),
  search_knowledge_base: t('searchTool.search_knowledge_base'),
};
/* eslint-enable camelcase -- keys are backend tool names, not camelCase identifiers */

export const isSearchStage = (value: string): value is SearchStage =>
  (SEARCH_STAGES as readonly string[]).includes(value);

const STAGE_LABELS: Record<SearchStage, string> = {
  context: t('searchStage.context'),
  contextualize: t('searchStage.contextualize'),
  generate: t('searchStage.generate'),
  rerank: t('searchStage.rerank'),
  retrieve: t('searchStage.retrieve'),
};

export const searchStageLabel = (stage: SearchStage): string =>
  STAGE_LABELS[stage];

export const searchToolIcon = (tool: string | undefined): LucideIcon =>
  tool === undefined ? Search : (TOOL_ICONS[tool] ?? Search);

// Tool-call rows: search tools keep their icon; data tools (courses, staff,
// timetables, ...) get a wrench so they do not read as searches.
export const toolCallIcon = (tool: string): LucideIcon =>
  TOOL_ICONS[tool] ?? (tool.includes('announcement') ? Megaphone : Wrench);

export const isSearchTool = (tool: string): boolean =>
  Object.hasOwn(TOOL_ICONS, tool);

export const searchToolLabel = (tool: string | undefined): string =>
  tool === undefined
    ? t('searchTool.fallback')
    : (TOOL_LABELS[tool] ?? t('searchTool.fallback'));

// Matched by name fragment in order (mirrors the API's table), so specific
// fragments come before general ones.
const DATA_TOOL_LABELS: ReadonlyArray<readonly [string, TKey]> = [
  ['recommend_diploma_committee', 'toolLabel.committee'],
  ['exam_session', 'toolLabel.examSessions'],
  ['timetable', 'toolLabel.timetable'],
  ['consultation', 'toolLabel.consultations'],
  ['weekly_topics', 'toolLabel.weeklyTopics'],
  ['study_program', 'toolLabel.studyPrograms'],
  ['academic_calendar', 'toolLabel.academicCalendar'],
  ['course', 'toolLabel.courses'],
  ['staff', 'toolLabel.staff'],
  ['room', 'toolLabel.rooms'],
  ['diploma', 'toolLabel.diplomas'],
  ['master', 'toolLabel.masters'],
  ['compan', 'toolLabel.companies'],
  ['quote', 'toolLabel.quote'],
];

// Tool rows outlive the call, so labels are tense-neutral noun phrases and the
// status icon carries running/done. The API's first-person `title` is meant for
// live status messages (Discord), so it is not shown here.
export const toolCallLabel = (tool: string): string => {
  if (isSearchTool(tool)) {
    return searchToolLabel(tool);
  }

  if (tool.includes('announcement')) {
    return t('toolLabel.announcements');
  }

  const match = DATA_TOOL_LABELS.find(([fragment]) => tool.includes(fragment));

  return t(match?.[1] ?? 'toolLabel.fallback');
};
