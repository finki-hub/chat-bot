export type ActionResult = boolean | undefined;

export type MaybeAsyncAction<Arguments extends readonly unknown[] = []> =
  | ((...arguments_: Arguments) => ActionResult | Promise<ActionResult>)
  | ((...arguments_: Arguments) => Promise<void>)
  | ((...arguments_: Arguments) => void);
