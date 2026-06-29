'use client';

import { useEffect, useState } from 'react';

export type Presence = {
  exiting: boolean;
  mounted: boolean;
};

const noop = (): void => {};

// Keeps content mounted through its exit transition: when `present` flips false
// the element stays mounted (with `exiting` set, so callers can apply "leaving"
// classes) until `exitMs` elapses, then unmounts. Flipping back to `present`
// cancels the pending unmount.
export const usePresence = (present: boolean, exitMs: number): Presence => {
  const [state, setState] = useState<Presence>(() => ({
    exiting: false,
    mounted: present,
  }));

  useEffect(() => {
    if (present) {
      setState({ exiting: false, mounted: true });
      return noop;
    }

    let cancelled = false;
    setState((prev) =>
      prev.mounted ? { exiting: true, mounted: true } : prev,
    );
    const id = setTimeout(() => {
      if (!cancelled) {
        setState({ exiting: false, mounted: false });
      }
    }, exitMs);

    return () => {
      cancelled = true;
      clearTimeout(id);
    };
  }, [present, exitMs]);

  return state;
};
