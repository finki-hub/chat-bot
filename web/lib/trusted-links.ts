// Faculty and university sites, and FINKI Hub itself: links to them open
// directly instead of behind the "open this link?" confirmation.
const TRUSTED_DOMAINS = [
  'finki.ukim.mk',
  'ukim.mk',
  'ukim.edu.mk',
  'finki-hub.com',
];

export const isTrustedLinkUrl = (value: string): boolean => {
  let url: URL;

  try {
    url = new URL(value);
  } catch {
    return false;
  }

  if (url.protocol !== 'https:') {
    return false;
  }

  const host = url.hostname.toLowerCase();

  return TRUSTED_DOMAINS.some(
    (domain) => host === domain || host.endsWith(`.${domain}`),
  );
};
