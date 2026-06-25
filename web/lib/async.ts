// Run a promise we intentionally don't await; surface any rejection to the
// global error handler.
export const fireAndForget = (promise: Promise<unknown>): void => {
  // A fire-and-forget helper must stay synchronous, so it cannot `await`; the
  // lone `.catch()` is the intended rejection sink, not promise-chaining to be
  // unwound into await.
  // eslint-disable-next-line promise/prefer-await-to-then, unicorn/prefer-await -- intentional rejection sink for a non-awaited promise
  promise.catch((error: unknown) => {
    reportError(error);
  });
};
