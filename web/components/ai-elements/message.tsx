"use client";

import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { t } from "@/lib/i18n";
import { cn } from "@/lib/utils";
import { cjk } from "@streamdown/cjk";
import { code } from "@streamdown/code";
import { math } from "@streamdown/math";
import { mermaid } from "@streamdown/mermaid";
import type { UIMessage } from "ai";
import type { ComponentProps, HTMLAttributes } from "react";
import { memo } from "react";
import {
  type LinkSafetyConfig,
  type LinkSafetyModalProps,
  Streamdown,
} from "streamdown";

export type MessageProps = HTMLAttributes<HTMLDivElement> & {
  from: UIMessage["role"];
};

export const Message = ({ className, from, ...props }: MessageProps) => (
  <div
    className={cn(
      "group flex w-full flex-col gap-2 motion-safe:animate-in motion-safe:fade-in-0 motion-safe:slide-in-from-bottom-2 motion-safe:duration-500",
      from === "user"
        ? "is-user ml-auto max-w-[85%] items-end"
        : "is-assistant max-w-full",
      className
    )}
    {...props}
  />
);

export type MessageContentProps = HTMLAttributes<HTMLDivElement>;

export const MessageContent = ({
  children,
  className,
  ...props
}: MessageContentProps) => (
  <div
    className={cn(
      "is-user:dark flex w-fit min-w-0 max-w-full flex-col gap-2 overflow-hidden text-sm leading-relaxed",
      "group-[.is-user]:ml-auto group-[.is-user]:rounded-2xl group-[.is-user]:rounded-br-md group-[.is-user]:bg-secondary group-[.is-user]:px-4 group-[.is-user]:py-2.5 group-[.is-user]:text-foreground group-[.is-user]:shadow-sm",
      "group-[.is-assistant]:text-foreground",
      className
    )}
    {...props}
  >
    {children}
  </div>
);

export type MessageResponseProps = ComponentProps<typeof Streamdown>;

const streamdownPlugins = { cjk, code, math, mermaid };

const markdownDomainLinkPattern =
  /(\[[^\]\n]+\]\(\s*)(?![a-z][a-z\d+.-]*:|[#/.])((?:www\.)?(?:[a-z\d](?:[a-z\d-]{0,61}[a-z\d])?\.)+[a-z\d](?:[a-z\d-]{0,61}[a-z\d])?(?:[/?#][^\s)]*)?)(\s*(?:["'][^"']*["']|\([^)]*\))?\))/giu;

const normalizeGeneratedMarkdownLinks = (content: string): string =>
  content.replace(
    markdownDomainLinkPattern,
    (_match, prefix: string, target: string, suffix: string) =>
      `${prefix}https://${target}${suffix}`
  );

const LinkSafetyModal = ({
  isOpen,
  onClose,
  onConfirm,
  url,
}: LinkSafetyModalProps) => (
  <ConfirmDialog
    confirmLabel={t("link.open")}
    description={
      <>
        {t("link.confirmDescription")}
        <span className="mt-2 block font-medium break-all text-foreground">
          {url}
        </span>
      </>
    }
    onConfirm={onConfirm}
    onOpenChange={(open) => {
      if (!open) {
        onClose();
      }
    }}
    open={isOpen}
    title={t("link.confirmTitle")}
  />
);

const linkSafety: LinkSafetyConfig = {
  enabled: true,
  renderModal: (props: LinkSafetyModalProps) => <LinkSafetyModal {...props} />,
};

export const MessageResponse = memo(
  ({ children, className, ...props }: MessageResponseProps) => {
    const normalizedChildren =
      typeof children === "string"
        ? normalizeGeneratedMarkdownLinks(children)
        : children;

    return (
      <Streamdown
        className={cn(
          "size-full [&>*:first-child]:mt-0 [&>*:last-child]:mb-0",
          className
        )}
        plugins={streamdownPlugins}
        linkSafety={linkSafety}
        {...props}
      >
        {normalizedChildren}
      </Streamdown>
    );
  },
  (prevProps, nextProps) =>
    prevProps.children === nextProps.children &&
    nextProps.isAnimating === prevProps.isAnimating
);

MessageResponse.displayName = "MessageResponse";
