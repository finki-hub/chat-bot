'use client';

import {
  messageRemarkPlugins,
  MessageResponse,
  type MessageResponseProps,
} from '@/components/ai-elements/message';
import { CitationRef } from '@/components/chat/citations';
import { CITATION_TAG, remarkCitations } from '@/lib/citations';

const remarkPlugins = [
  ...messageRemarkPlugins,
  remarkCitations,
] satisfies NonNullable<MessageResponseProps['remarkPlugins']>;
// hast property name, which React receives as `data-ref`.
const allowedTags = { [CITATION_TAG]: ['dataRef'] };
// `cite-ref` is a custom tag, outside the intrinsic-element keys Components lists.
const components = {
  [CITATION_TAG]: CitationRef,
} as NonNullable<MessageResponseProps['components']>;

// An assistant answer: Markdown whose `[N]` markers become source chips.
export const AnswerMarkdown = ({
  children,
  ...props
}: Omit<
  MessageResponseProps,
  'allowedTags' | 'components' | 'remarkPlugins'
>) => (
  <MessageResponse
    {...props}
    allowedTags={allowedTags}
    // Streamdown pads every list item; long answers read better a bit tighter.
    className={`[&_li]:py-0.5 ${props.className ?? ''}`}
    components={components}
    remarkPlugins={remarkPlugins}
  >
    {children}
  </MessageResponse>
);
