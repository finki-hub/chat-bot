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
  defaultRemarkPlugins,
} from "streamdown";

export type MessageProps = HTMLAttributes<HTMLDivElement> & {
  from: UIMessage["role"];
};

export const Message = ({ className, from, ...props }: MessageProps) => (
  <div
    className={cn(
      "group flex w-full flex-col gap-2",
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

type MarkdownNode = {
  children?: unknown;
  type?: unknown;
  url?: unknown;
};

const urlSchemePattern = /^[a-z][a-z\d+.-]*:/iu;
const domainUrlPattern =
  /^(?:www\.)?(?:[a-z\d](?:[a-z\d-]{0,61}[a-z\d])?\.)+[a-z\d](?:[a-z\d-]{0,61}[a-z\d])?(?:[/?#].*)?$/iu;

const isMarkdownNode = (value: unknown): value is MarkdownNode =>
  typeof value === "object" && value !== null;

const normalizeMarkdownLinkTarget = (url: string): string => {
  if (
    urlSchemePattern.test(url) ||
    url.startsWith("#") ||
    url.startsWith("/") ||
    url.startsWith(".") ||
    !domainUrlPattern.test(url)
  ) {
    return url;
  }

  return `https://${url}`;
};

const normalizeMarkdownLinks = (node: unknown): void => {
  if (!isMarkdownNode(node)) {
    return;
  }

  if (node.type === "link" && typeof node.url === "string") {
    node.url = normalizeMarkdownLinkTarget(node.url);
  }

  if (Array.isArray(node.children)) {
    for (const child of node.children) {
      normalizeMarkdownLinks(child);
    }
  }
};

const normalizeGeneratedMarkdownLinks = () => normalizeMarkdownLinks;

const remarkPlugins = [
  ...Object.values(defaultRemarkPlugins),
  normalizeGeneratedMarkdownLinks,
] satisfies NonNullable<MessageResponseProps["remarkPlugins"]>;

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
  ({ className, ...props }: MessageResponseProps) => (
    <Streamdown
      className={cn(
        "size-full [&>*:first-child]:mt-0 [&>*:last-child]:mb-0",
        className
      )}
      plugins={streamdownPlugins}
      linkSafety={linkSafety}
      remarkPlugins={remarkPlugins}
      {...props}
    />
  ),
  (prevProps, nextProps) =>
    prevProps.children === nextProps.children &&
    nextProps.isAnimating === prevProps.isAnimating
);

MessageResponse.displayName = "MessageResponse";
