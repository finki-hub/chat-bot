import { fireEvent, screen, within } from '@testing-library/react';

// Opens a sidebar row's actions menu and picks one of its items.
export const chooseRowAction = (row: HTMLElement, label: string): void => {
  fireEvent.keyDown(within(row).getByTestId('row-actions'), { key: 'Enter' });
  fireEvent.click(screen.getByRole('menuitem', { name: label }));
};
