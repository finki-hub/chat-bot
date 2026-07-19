import type { ReactNode } from 'react';

import { render, screen } from '@testing-library/react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';

import ErrorPage from '@/app/error';
import RootLayout from '@/app/layout';
import NotFound from '@/app/not-found';
import { ComposerActions } from '@/components/chat/composer-actions';
import { CredentialSettingsStatus } from '@/components/shell/credential-settings-status';
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogTitle,
} from '@/components/ui/dialog';
import { IconButton } from '@/components/ui/icon-controls';
import { InputGroupButton } from '@/components/ui/input-group';

vi.mock('next/script', () => ({
  default: ({ children }: { readonly children: ReactNode }) => children,
}));

vi.mock('@/app/providers', () => ({
  Providers: ({ children }: { readonly children: ReactNode }) => children,
}));

describe('root navigation', () => {
  it('renders a localized skip link before page content', () => {
    const markup = renderToStaticMarkup(
      <RootLayout>
        <main id="main-content">Содржина</main>
      </RootLayout>,
    );

    expect(markup).toContain('href="#main-content"');
    expect(markup).toContain('Прескокни до содржината');
  });

  it('provides a stable skip target on fallback pages', () => {
    const notFoundMarkup = renderToStaticMarkup(<NotFound />);
    const errorMarkup = renderToStaticMarkup(
      <ErrorPage
        error={new Error('test')}
        reset={vi.fn<() => void>()}
      />,
    );

    expect(notFoundMarkup).toContain('id="main-content"');
    expect(errorMarkup).toContain('id="main-content"');
  });
});

describe('coarse-pointer targets', () => {
  it('keeps shared compact controls at least 44px on coarse pointers', () => {
    render(
      <>
        <IconButton aria-label="Икона" />
        <InputGroupButton
          aria-label="Исчисти"
          size="icon-xs"
        />
        <InputGroupButton
          aria-label="Текстуална акција"
          size="xs"
        />
      </>,
    );

    expect(screen.getByRole('button', { name: 'Икона' })).toHaveClass(
      'sm:pointer-fine:size-9',
    );
    expect(screen.getByRole('button', { name: 'Исчисти' })).toHaveClass(
      'pointer-coarse:size-11',
    );
    expect(
      screen.getByRole('button', { name: 'Текстуална акција' }),
    ).toHaveClass('pointer-coarse:h-11');
  });

  it('compacts composer selectors only for fine pointers', () => {
    render(
      <ComposerActions
        availableProviders={new Set()}
        groups={[]}
        isBusy={false}
        model="model-a"
        modelPlaceholder="Модел"
        modelSelectDisabled={false}
        onButtonClick={vi.fn<() => void>()}
        onModelChange={vi.fn<(model: string) => void>()}
        onReasoningChange={vi.fn<(reasoning: boolean) => void>()}
        reasoning={false}
        showModelPlaceholder
        status="ready"
        submitDisabled={false}
      />,
    );

    expect(screen.getByTestId('composer-reasoning')).toHaveClass(
      'sm:pointer-fine:min-h-8',
    );
    expect(screen.getByTestId('composer-model')).toHaveClass(
      'sm:pointer-fine:min-h-8',
    );
  });

  it('gives dialog close and footer actions coarse-pointer targets', () => {
    render(
      <Dialog open>
        <DialogContent>
          <DialogTitle>Потврда</DialogTitle>
          <DialogFooter data-testid="dialog-footer">
            <button type="button">Продолжи</button>
          </DialogFooter>
        </DialogContent>
      </Dialog>,
    );

    expect(screen.getByRole('button', { name: 'Затвори' })).toHaveClass(
      'pointer-coarse:size-11',
    );
    expect(screen.getByTestId('dialog-footer')).toHaveClass(
      'pointer-coarse:[&>button]:min-h-11',
    );
  });

  it('gives credential recovery actions coarse-pointer targets', () => {
    render(
      <CredentialSettingsStatus
        loadError
        loading={false}
        onRetryAction={vi.fn<() => void>()}
      />,
    );

    expect(
      screen.getByRole('button', { name: 'Обиди се повторно' }),
    ).toHaveClass('pointer-coarse:min-h-11');
  });
});
