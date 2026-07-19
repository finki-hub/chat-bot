"use client";

import { Button } from "@/components/ui/button";
import { t } from "@/lib/i18n";
import { cn } from "@/lib/utils";
import { ArrowDownIcon } from "lucide-react";
import type { ComponentProps } from "react";
import { useCallback, useEffect, useState } from "react";
import { StickToBottom, useStickToBottomContext } from "use-stick-to-bottom";

const REDUCED_MOTION_QUERY = '(prefers-reduced-motion: reduce)';

const prefersReducedMotion = () =>
  typeof matchMedia === 'function' && matchMedia(REDUCED_MOTION_QUERY).matches;

const usePrefersReducedMotion = () => {
  const [reducedMotion, setReducedMotion] = useState(prefersReducedMotion);

  useEffect(() => {
    if (typeof matchMedia !== 'function') {
      return;
    }

    const media = matchMedia(REDUCED_MOTION_QUERY);
    const handleChange = (event: MediaQueryListEvent) => {
      setReducedMotion(event.matches);
    };

    setReducedMotion(media.matches);
    media.addEventListener('change', handleChange);

    return () => {
      media.removeEventListener('change', handleChange);
    };
  }, []);

  return reducedMotion;
};

export type ConversationProps = ComponentProps<typeof StickToBottom>;

export const Conversation = ({ className, ...props }: ConversationProps) => {
  const reducedMotion = usePrefersReducedMotion();
  const scrollBehavior = reducedMotion ? 'instant' : 'smooth';

  return (
    <StickToBottom
      className={cn("relative flex-1 overflow-y-hidden", className)}
      initial={scrollBehavior}
      resize={scrollBehavior}
      role="log"
      {...props}
    />
  );
};

export type ConversationContentProps = ComponentProps<
  typeof StickToBottom.Content
>;

export const ConversationContent = ({
  className,
  ...props
}: ConversationContentProps) => (
  <StickToBottom.Content
    className={cn("flex min-h-full flex-col px-4 py-6", className)}
    {...props}
  />
);

export type ConversationScrollButtonProps = ComponentProps<typeof Button>;

export const ConversationScrollButton = ({
  className,
  ...props
}: ConversationScrollButtonProps) => {
  const { isAtBottom, scrollToBottom } = useStickToBottomContext();
  const reducedMotion = usePrefersReducedMotion();

  const handleScrollToBottom = useCallback(() => {
    if (reducedMotion) {
      scrollToBottom('instant');
      return;
    }

    scrollToBottom();
  }, [reducedMotion, scrollToBottom]);

  return (
    !isAtBottom && (
      <Button
        aria-label={t('conversation.scrollToLatest')}
        className={cn(
          "absolute bottom-4 left-[50%] translate-x-[-50%] rounded-full pointer-coarse:size-11 dark:bg-background dark:hover:bg-muted",
          className
        )}
        onClick={handleScrollToBottom}
        size="icon"
        type="button"
        variant="outline"
        {...props}
      >
        <ArrowDownIcon
          aria-hidden="true"
          className="size-4"
        />
      </Button>
    )
  );
};
