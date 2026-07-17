import type {
  AnchorHTMLAttributes,
  ButtonHTMLAttributes,
  ReactElement,
} from 'react';

import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { cn } from '@/lib/utils';

const iconControlClass =
  'inline-flex size-11 cursor-pointer items-center justify-center rounded-md border border-input bg-background text-sm font-medium ring-offset-background transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none sm:size-9';

const ControlTooltip = ({
  children,
  disabled = false,
  label,
}: {
  readonly children: ReactElement;
  readonly disabled?: boolean;
  readonly label: string;
}) => (
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

const IconButton = ({
  className,
  disabled = false,
  title,
  type = 'button',
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement>) => {
  const control = (
    <button
      className={cn(iconControlClass, className)}
      disabled={disabled}
      type={type}
      {...props}
    />
  );

  return title ? (
    <ControlTooltip
      disabled={disabled}
      label={title}
    >
      {control}
    </ControlTooltip>
  ) : (
    control
  );
};

const IconLink = ({
  className,
  title,
  ...props
}: AnchorHTMLAttributes<HTMLAnchorElement>) => {
  const control = (
    <a
      aria-label={title}
      className={cn(iconControlClass, className)}
      {...props}
    />
  );

  return title ? (
    <ControlTooltip label={title}>{control}</ControlTooltip>
  ) : (
    control
  );
};

export { IconButton, IconLink };
