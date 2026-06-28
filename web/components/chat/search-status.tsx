import { searchToolIcon, searchToolLabel } from '@/lib/search-status';

export type SearchStatusProps = {
  label?: string;
  tool?: string;
};

export const SearchStatus = ({ label, tool }: SearchStatusProps) => {
  const Icon = searchToolIcon(tool);
  const text = label ?? searchToolLabel(tool);

  return (
    <output
      aria-live="polite"
      className="inline-flex items-center gap-2 rounded-full border border-border bg-muted/60 px-3 py-1 text-sm text-muted-foreground motion-safe:animate-in motion-safe:fade-in-0 motion-safe:zoom-in-95"
      data-testid="search-status"
      data-tool={tool}
    >
      <Icon
        aria-hidden="true"
        className="size-4 animate-spin"
      />
      <span>{text}</span>
    </output>
  );
};
