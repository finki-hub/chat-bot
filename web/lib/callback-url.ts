const fallbackCallbackUrl = '/';

export const getSafeCallbackUrl = (callbackUrl: string | undefined): string => {
  if (callbackUrl === undefined) {
    return fallbackCallbackUrl;
  }

  const isSameOriginRelative =
    callbackUrl.startsWith('/') &&
    !callbackUrl.startsWith('//') &&
    !callbackUrl.startsWith('/\\');

  return isSameOriginRelative ? callbackUrl : fallbackCallbackUrl;
};
