import type { ReactElement } from 'react';

import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';

type ConversationActionTooltipProps = {
  readonly children: ReactElement;
  readonly disabled?: boolean;
  readonly label: string;
};

export const ConversationActionTooltip = ({
  children,
  disabled = false,
  label,
}: ConversationActionTooltipProps) => (
  <TooltipProvider>
    <Tooltip disableHoverableContent>
      <TooltipTrigger asChild>
        {disabled ? <span className="inline-flex">{children}</span> : children}
      </TooltipTrigger>
      <TooltipContent
        side="bottom"
        sideOffset={4}
      >
        {label}
      </TooltipContent>
    </Tooltip>
  </TooltipProvider>
);
