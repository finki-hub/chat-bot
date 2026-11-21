import { t } from '@/lib/i18n';

const STEPS = [
  t('committee.nextSteps.contact'),
  t('committee.nextSteps.agree'),
  t('committee.nextSteps.approve'),
];

export const NextSteps = () => (
  <section className="rounded-xl border border-border bg-card p-4">
    <h3 className="mb-3 text-sm font-semibold text-foreground">
      {t('committee.nextSteps.title')}
    </h3>
    <ol className="space-y-2">
      {STEPS.map((step, index) => (
        <li
          className="flex items-start gap-2.5 text-sm text-muted-foreground"
          key={step}
        >
          <span className="flex size-5 shrink-0 items-center justify-center rounded-full bg-primary/10 text-xs font-semibold text-primary">
            {index + 1}
          </span>
          {step}
        </li>
      ))}
    </ol>
  </section>
);
