import '@testing-library/jest-dom/vitest';
import 'fake-indexeddb/auto';
import { vi } from 'vitest';

import { ResizeObserverStub } from '@/test/helpers/dom-stubs';

const noop = (): void => {
  // jsdom stub for DOM methods Radix UI (Select, etc.) calls but jsdom omits.
};

vi.stubGlobal('ResizeObserver', ResizeObserverStub);
/* eslint-disable sonarjs/class-prototype -- stub prototype methods jsdom does not implement */
Element.prototype.scrollIntoView = noop;
Element.prototype.setPointerCapture = noop;
Element.prototype.releasePointerCapture = noop;
Element.prototype.hasPointerCapture = () => false;
/* eslint-enable sonarjs/class-prototype -- restore the rule after the stubs above */
