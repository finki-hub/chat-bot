export const fireAndForget = (promise: Promise<unknown>): void => {
  // eslint-disable-next-line promise/prefer-await-to-then, unicorn/prefer-await -- intentional rejection sink for a non-awaited promise
  promise.catch((error: unknown) => {
    reportError(error);
  });
};
