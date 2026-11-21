import { t } from '@/lib/i18n';

export const formatDefenseCount = (count: number): string =>
  `${count} ${t(count === 1 ? 'committee.why.defenseOne' : 'committee.why.defenseMany')}`;

export const formatPaperCount = (count: number): string =>
  `${count} ${t(count === 1 ? 'committee.why.paperOne' : 'committee.why.paperMany')}`;
