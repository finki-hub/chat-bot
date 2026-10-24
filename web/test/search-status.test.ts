import { FileText, Megaphone, Search, Wrench } from 'lucide-react';
import { describe, expect, it } from 'vitest';

import { isSearchTool, toolCallIcon } from '@/lib/search-status';

describe('toolCallIcon', () => {
  it('keeps search icons and marks announcements and data tools apart', () => {
    expect(toolCallIcon('search_knowledge_base')).toBe(Search);
    expect(toolCallIcon('search_documents')).toBe(FileText);
    expect(toolCallIcon('search_announcements')).toBe(Megaphone);
    expect(toolCallIcon('get_announcement')).toBe(Megaphone);
    expect(toolCallIcon('list_courses')).toBe(Wrench);
  });

  it('does not count announcement tools as knowledge-base searches', () => {
    expect(isSearchTool('search_announcements')).toBe(false);
  });
});
