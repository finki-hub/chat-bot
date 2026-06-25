import { Loader2 } from 'lucide-react';

export type SearchStatusProps = {
  label: string;
  tool?: string;
};

export const SearchStatus = ({ label, tool }: SearchStatusProps) => (
  <output
    aria-live="polite"
    className="inline-flex items-center gap-2 rounded-full border border-border bg-muted/60 px-3 py-1 text-sm text-muted-foreground"
    data-testid="search-status"
    data-tool={tool}
  >
    <Loader2
      aria-hidden="true"
      className="size-4 animate-spin"
    />
    <span>{label}</span>
  </output>
);
