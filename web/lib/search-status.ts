import {
  FileText,
  type LucideIcon,
  MessageCircleQuestion,
  Search,
} from 'lucide-react';

import { t } from '@/lib/i18n';

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
};

const TOOL_LABELS: Record<string, string> = {
  faq_search: t('searchTool.faq_search'),
  search_docs: t('searchTool.search_docs'),
  search_documents: t('searchTool.search_documents'),
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

export const searchToolLabel = (tool: string | undefined): string =>
  tool === undefined
    ? t('searchTool.fallback')
    : (TOOL_LABELS[tool] ?? t('searchTool.fallback'));
