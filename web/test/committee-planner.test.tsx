/* eslint-disable camelcase -- API response fixtures use snake_case property names */
import type { ReactNode } from 'react';

import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  type Mock,
  vi,
} from 'vitest';

import type { CommitteeResponse, Professor } from '@/lib/committee-types';

import { CommitteePlanner } from '@/components/committee/committee-planner';
import { t } from '@/lib/i18n';
import { ResizeObserverStub } from '@/test/helpers/dom-stubs';

const TITLE_LABEL = t('committee.titleField');
const ABSTRACT_LABEL = t('committee.abstract');
const MENTOR_GIVEN_LABEL = t('committee.mentorMode.given');
const LOADING_TEXT = t('committee.loadingStaff');

const MENTOR_BOJAN = 'Бојан Илиевски';
const MENTOR_VLADIMIR = 'Владимир Трајковиќ';
const OTHER_MENTOR = 'Друг Професор';
const PROFESSOR_NAME = 'Ана Богоевска';
const PROFESSOR_NAME_PATTERN = new RegExp(PROFESSOR_NAME, 'u');
const SECRETARY_NAME_PATTERN = /Секретар Тест/u;
const TITLE_DIPLOMA = 'Систем за препорака на ментори';
const TITLE_THESIS = 'Тема за дипломска';
const TITLE_SIMPLE = 'Тема';
const ABSTRACT_TEXT = 'Апстракт текст';
const KEYWORD_ML = 'машинско учење';
const KEYWORD_NLP = 'NLP';
const PROGRAM_KI = 'КИ';
const AREA_NLP = 'NLP';
const TITLE_DR = 'д-р';
const URL_PROFESSORS = '/api/professors';
const URL_COMMITTEE = '/api/committee';
const TESTID_SUBMIT = 'committee-submit';
const TESTID_MENTOR_SELECT = 'committee-mentor-select';
const TESTID_ERROR = 'committee-error';
const TESTID_RESULT = 'committee-result';

vi.mock('next/link', () => ({
  default: ({
    children,
    href,
    ...props
  }: {
    readonly children: ReactNode;
    readonly href: string;
  }) => (
    <a
      href={href}
      {...props}
    >
      {children}
    </a>
  ),
}));

const PROFESSORS: Professor[] = [
  {
    cabinet: '226',
    consultations: 'https://consultations.finki.ukim.mk/display/ana',
    id: 1,
    mentor_eligible: true,
    name: PROFESSOR_NAME,
    photo: null,
    position: 'Доцент',
    profile: 'https://finki.ukim.mk/ana',
    title: TITLE_DR,
  },
  {
    cabinet: null,
    consultations: null,
    id: 2,
    mentor_eligible: true,
    name: MENTOR_BOJAN,
    photo: null,
    position: 'Професор',
    profile: null,
    title: 'проф. д-р',
  },
  {
    cabinet: null,
    consultations: null,
    id: 3,
    mentor_eligible: true,
    name: MENTOR_VLADIMIR,
    photo: null,
    position: 'Доцент',
    profile: null,
    title: TITLE_DR,
  },
];

const createSuccessResponse = (): CommitteeResponse => ({
  alternatives: [
    {
      members: [MENTOR_BOJAN, MENTOR_VLADIMIR],
      mentor: PROFESSOR_NAME,
      score: 0.75,
      supporting_diploma_ids: ['d2'],
    },
  ],
  confidence: 0.82,
  confidence_reasons: ['strong_topic_match', 'enough_similar_defenses'],
  evidence: {
    similar_diplomas: [
      {
        external_id: '2023-001',
        id: 'd1',
        members: [MENTOR_BOJAN],
        mentor: PROFESSOR_NAME,
        rerank_score: 0.9,
        similarity: 0.85,
        title: TITLE_DIPLOMA,
      },
    ],
    supporting_paper_titles: ['Deep learning за препораки'],
  },
  members: [
    {
      buddy_score: 0.7,
      defense_score: 0.8,
      expertise_score: 0.9,
      name: MENTOR_BOJAN,
      prior_score: 0,
      score: 0.85,
      supporting_diploma_ids: ['d1'],
      supporting_paper_titles: [],
    },
    {
      buddy_score: 0.6,
      defense_score: 0.75,
      expertise_score: 0.8,
      name: MENTOR_VLADIMIR,
      prior_score: 0,
      score: 0.78,
      supporting_diploma_ids: [],
      supporting_paper_titles: [],
    },
  ],
  mentor: {
    buddy_score: 0.8,
    defense_score: 0.9,
    expertise_score: 0.95,
    name: PROFESSOR_NAME,
    prior_score: 0,
    score: 0.92,
    supporting_diploma_ids: ['d1'],
    supporting_paper_titles: ['Deep learning за препораки'],
  },
  mentor_is_given: true,
  mode: 'members_only',
  supporting_diploma_ids: ['d1'],
});

const getFetchUrl = (input: RequestInfo | URL): string => {
  if (typeof input === 'string') return input;
  if (input instanceof URL) return input.href;
  return input.url;
};

const setupFetchMock = (
  committeeResponse?: CommitteeResponse | Response,
  professors: Professor[] = PROFESSORS,
): Mock<typeof fetch> => {
  const fetchMock = vi.fn<typeof fetch>();

  fetchMock.mockImplementation((input) => {
    const url = getFetchUrl(input);

    if (url.includes(URL_PROFESSORS)) {
      return Promise.resolve(Response.json(professors));
    }

    if (url.includes(URL_COMMITTEE)) {
      const response =
        committeeResponse instanceof Response
          ? committeeResponse
          : Response.json(committeeResponse ?? createSuccessResponse());
      return Promise.resolve(response);
    }

    return Promise.resolve(new Response('not found', { status: 404 }));
  });

  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
};

const waitForProfessors = async () => {
  await waitFor(() => {
    expect(screen.queryByText(LOADING_TEXT)).not.toBeInTheDocument();
  });
};

const submitTitle = async (user: ReturnType<typeof userEvent.setup>) => {
  await waitForProfessors();
  await user.type(screen.getByLabelText(TITLE_LABEL), TITLE_SIMPLE);
  await user.click(screen.getByTestId(TESTID_SUBMIT));
  await screen.findByTestId(TESTID_RESULT);
};

describe('CommitteePlanner', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn<typeof fetch>());
    vi.stubGlobal('ResizeObserver', ResizeObserverStub);
  });

  afterEach(() => {
    cleanup();
    localStorage.clear();
    vi.unstubAllGlobals();
  });

  it('requires a title before submitting', async () => {
    setupFetchMock();
    const user = userEvent.setup();
    render(<CommitteePlanner />);

    await waitForProfessors();
    await user.click(screen.getByTestId(TESTID_SUBMIT));

    expect(screen.getByLabelText(TITLE_LABEL)).toHaveAttribute(
      'aria-invalid',
      'true',
    );
    expect(screen.getByText(t('committee.error.titleRequired'))).toBeVisible();
    expect(vi.mocked(fetch)).not.toHaveBeenCalledWith(
      expect.stringContaining(URL_COMMITTEE),
      expect.anything(),
    );
  });

  it('requires a mentor when mentor mode is set to given', async () => {
    setupFetchMock();
    const user = userEvent.setup();
    render(<CommitteePlanner />);

    await waitForProfessors();
    await user.click(screen.getByLabelText(MENTOR_GIVEN_LABEL));
    await user.type(screen.getByLabelText(TITLE_LABEL), TITLE_SIMPLE);
    await user.click(screen.getByTestId(TESTID_SUBMIT));

    expect(screen.getByTestId(TESTID_MENTOR_SELECT)).toHaveAttribute(
      'aria-invalid',
      'true',
    );
    expect(screen.getByText(t('committee.error.mentorRequired'))).toBeVisible();
    expect(
      screen.queryByText(t('committee.error.unknownMentor')),
    ).not.toBeInTheDocument();
    expect(vi.mocked(fetch)).not.toHaveBeenCalledWith(
      expect.stringContaining(URL_COMMITTEE),
      expect.anything(),
    );
  });

  it('sends include_prior false and the chosen mentor', async () => {
    setupFetchMock();
    const user = userEvent.setup();
    render(<CommitteePlanner />);

    await waitForProfessors();
    await user.click(screen.getByLabelText(MENTOR_GIVEN_LABEL));
    await user.type(screen.getByLabelText(TITLE_LABEL), TITLE_THESIS);
    await user.type(screen.getByLabelText(ABSTRACT_LABEL), ABSTRACT_TEXT);
    await user.type(
      screen.getByLabelText(t('committee.researchArea')),
      AREA_NLP,
    );
    await user.type(
      screen.getByLabelText(t('committee.studyProgram')),
      PROGRAM_KI,
    );

    const keywordsInput = screen.getByTestId('committee-keywords-input');
    await user.type(keywordsInput, KEYWORD_ML);
    await user.keyboard('{Enter}');
    await user.type(keywordsInput, KEYWORD_NLP);
    await user.keyboard('{Enter}');

    await user.click(screen.getByTestId(TESTID_MENTOR_SELECT));
    fireEvent.click(
      screen.getByRole('option', { name: PROFESSOR_NAME_PATTERN }),
    );

    await user.click(screen.getByTestId(TESTID_SUBMIT));

    await waitFor(() => {
      expect(vi.mocked(fetch)).toHaveBeenCalledWith(
        URL_COMMITTEE,
        expect.objectContaining({
          method: 'POST',
        }),
      );
    });

    const committeeCalls = vi
      .mocked(fetch)
      .mock.calls.filter(
        ([url]) => typeof url === 'string' && url.includes(URL_COMMITTEE),
      );

    expect(committeeCalls).toHaveLength(1);

    const call = committeeCalls[0];
    if (call === undefined) throw new Error('Missing committee call');
    const requestInit = call[1];
    if (requestInit === undefined) throw new Error('Missing request init');
    const body = JSON.parse(requestInit.body as string) as Record<
      string,
      unknown
    >;

    expect(body).toStrictEqual({
      abstract: ABSTRACT_TEXT,
      alternatives: 3,
      include_prior: false,
      keywords: [KEYWORD_ML, KEYWORD_NLP.toLowerCase()],
      mentor: PROFESSOR_NAME,
      research_area: AREA_NLP,
      study_program: PROGRAM_KI,
      title: TITLE_THESIS,
    });
  });

  it('renders committee names, evidence, and disclaimer without numeric scores', async () => {
    setupFetchMock(createSuccessResponse());
    const user = userEvent.setup();
    render(<CommitteePlanner />);

    await waitForProfessors();
    await user.type(screen.getByLabelText(TITLE_LABEL), TITLE_SIMPLE);
    await user.click(screen.getByTestId(TESTID_SUBMIT));

    await waitFor(() => {
      expect(screen.getByTestId(TESTID_RESULT)).toBeInTheDocument();
    });

    expect(screen.getByText(PROFESSOR_NAME)).toBeVisible();
    expect(screen.getByText(MENTOR_BOJAN)).toBeVisible();
    expect(screen.getByText(MENTOR_VLADIMIR)).toBeVisible();
    expect(screen.getByText(TITLE_DIPLOMA)).toBeVisible();
    expect(screen.getByText(t('committee.disclaimer'))).toBeVisible();
    expect(
      screen.getByRole('link', { name: t('committee.consultations') }),
    ).toHaveAttribute(
      'href',
      'https://consultations.finki.ukim.mk/display/ana',
    );

    const result = screen.getByTestId(TESTID_RESULT);

    expect(result).not.toHaveTextContent('0.92');
    expect(result).not.toHaveTextContent('0.85');
    expect(result).not.toHaveTextContent('0.78');
    expect(result).not.toHaveTextContent('0.82');
  });

  it('summarizes confidence in one plain message', async () => {
    setupFetchMock(createSuccessResponse());
    const user = userEvent.setup();
    render(<CommitteePlanner />);

    await waitForProfessors();
    await user.type(screen.getByLabelText(TITLE_LABEL), TITLE_SIMPLE);
    await user.click(screen.getByTestId(TESTID_SUBMIT));

    await waitFor(() => {
      expect(screen.getByTestId(TESTID_RESULT)).toBeInTheDocument();
    });

    expect(screen.getByTestId('committee-confidence')).toHaveTextContent(
      t('committee.confidence.high'),
    );
  });

  it('shows a 404 state when there are not enough similar defenses', async () => {
    setupFetchMock(
      Response.json({ detail: 'No similar defenses' }, { status: 404 }),
    );
    const user = userEvent.setup();
    render(<CommitteePlanner />);

    await waitForProfessors();
    await user.type(screen.getByLabelText(TITLE_LABEL), TITLE_SIMPLE);
    await user.click(screen.getByTestId(TESTID_SUBMIT));

    await waitFor(() => {
      expect(screen.getByTestId(TESTID_ERROR)).toHaveTextContent(
        t('committee.error.notFound'),
      );
    });
  });

  it('shows an upstream 422 detail and points mentor errors at the mentor field', async () => {
    setupFetchMock(
      Response.json(
        { detail: 'mentor must be an active staff member' },
        { status: 422 },
      ),
    );
    const user = userEvent.setup();
    render(<CommitteePlanner />);

    await waitForProfessors();
    await user.click(screen.getByLabelText(MENTOR_GIVEN_LABEL));
    await user.type(screen.getByLabelText(TITLE_LABEL), TITLE_SIMPLE);
    await user.click(screen.getByTestId(TESTID_MENTOR_SELECT));
    fireEvent.click(
      screen.getByRole('option', { name: PROFESSOR_NAME_PATTERN }),
    );
    await user.click(screen.getByTestId(TESTID_SUBMIT));

    await waitFor(() => {
      expect(screen.getByTestId(TESTID_ERROR)).toHaveTextContent(
        t('committee.error.unknownMentor'),
      );
    });

    expect(screen.getByTestId(TESTID_MENTOR_SELECT)).toHaveAttribute(
      'aria-invalid',
      'true',
    );
  });

  it('shows a 429 state with retry guidance', async () => {
    setupFetchMock(
      Response.json({ error: 'Too many requests' }, { status: 429 }),
    );
    const user = userEvent.setup();
    render(<CommitteePlanner />);

    await waitForProfessors();
    await user.type(screen.getByLabelText(TITLE_LABEL), TITLE_SIMPLE);
    await user.click(screen.getByTestId(TESTID_SUBMIT));

    await waitFor(() => {
      expect(screen.getByTestId(TESTID_ERROR)).toHaveTextContent(
        t('committee.error.tooManyRequests'),
      );
    });
  });

  it('shows a 503/network error state with retry guidance', async () => {
    setupFetchMock(Response.json({ error: 'Unavailable' }, { status: 503 }));
    const user = userEvent.setup();
    render(<CommitteePlanner />);

    await waitForProfessors();
    await user.type(screen.getByLabelText(TITLE_LABEL), TITLE_SIMPLE);
    await user.click(screen.getByTestId(TESTID_SUBMIT));

    await waitFor(() => {
      expect(screen.getByTestId(TESTID_ERROR)).toHaveTextContent(
        t('committee.error.networkError'),
      );
    });
  });

  it('shows an empty committee message when the response is degenerate', async () => {
    const degenerateResponse: CommitteeResponse = {
      alternatives: [],
      confidence: 0,
      confidence_reasons: [],
      evidence: { similar_diplomas: [], supporting_paper_titles: [] },
      members: [],
      mentor: {
        buddy_score: 0,
        defense_score: 0,
        expertise_score: 0,
        name: '',
        prior_score: 0,
        score: 0,
        supporting_diploma_ids: [],
        supporting_paper_titles: [],
      },
      mentor_is_given: false,
      mode: 'full',
      supporting_diploma_ids: [],
    };

    setupFetchMock(degenerateResponse);
    const user = userEvent.setup();
    render(<CommitteePlanner />);

    await waitForProfessors();
    await user.type(screen.getByLabelText(TITLE_LABEL), TITLE_SIMPLE);
    await user.click(screen.getByTestId(TESTID_SUBMIT));

    await waitFor(() => {
      expect(screen.getByTestId(TESTID_ERROR)).toHaveTextContent(
        t('committee.error.emptyCommittee'),
      );
    });
  });

  it('hides degenerate alternatives', async () => {
    const response = createSuccessResponse();
    response.mode = 'full';
    response.alternatives = [
      {
        members: [MENTOR_BOJAN],
        mentor: null,
        score: 0.5,
        supporting_diploma_ids: [],
      },
      {
        members: [],
        mentor: MENTOR_VLADIMIR,
        score: 0,
        supporting_diploma_ids: [],
      },
    ];
    setupFetchMock(response);
    const user = userEvent.setup();
    render(<CommitteePlanner />);

    await waitForProfessors();
    await user.type(screen.getByLabelText(TITLE_LABEL), TITLE_SIMPLE);
    await user.click(screen.getByTestId(TESTID_SUBMIT));

    await waitFor(() => {
      expect(screen.getByTestId(TESTID_RESULT)).toBeInTheDocument();
    });

    expect(
      screen.queryByText(t('committee.result.alternativeConsider')),
    ).not.toBeInTheDocument();
  });

  it('includes a keyword typed but not yet committed', async () => {
    setupFetchMock();
    const user = userEvent.setup();
    render(<CommitteePlanner />);

    await waitForProfessors();
    await user.type(screen.getByLabelText(TITLE_LABEL), TITLE_SIMPLE);
    await user.type(
      screen.getByTestId('committee-keywords-input'),
      'граф теорија',
    );
    await user.click(screen.getByTestId(TESTID_SUBMIT));

    await waitFor(() => {
      expect(vi.mocked(fetch)).toHaveBeenCalledWith(
        URL_COMMITTEE,
        expect.objectContaining({ method: 'POST' }),
      );
    });

    const call = vi
      .mocked(fetch)
      .mock.calls.find(([url]) => url === URL_COMMITTEE);
    if (call === undefined) throw new Error('Missing committee call');
    const rawBody = call[1]?.body;
    if (typeof rawBody !== 'string') {
      throw new TypeError('Missing request body');
    }
    const body = JSON.parse(rawBody) as { keywords?: string[] };

    expect(body.keywords).toStrictEqual(['граф теорија']);
  });

  it('shows a generic validation message for an array detail', async () => {
    setupFetchMock(
      Response.json({ detail: [{ msg: 'field required' }] }, { status: 422 }),
    );
    const user = userEvent.setup();
    render(<CommitteePlanner />);

    await waitForProfessors();
    await user.type(screen.getByLabelText(TITLE_LABEL), TITLE_SIMPLE);
    await user.click(screen.getByTestId(TESTID_SUBMIT));

    await waitFor(() => {
      expect(screen.getByTestId(TESTID_ERROR)).toHaveTextContent(
        t('committee.error.invalidRequest'),
      );
    });
  });

  it('offers only mentor-eligible staff in the picker', async () => {
    const ineligible: Professor = {
      cabinet: null,
      consultations: null,
      id: 99,
      mentor_eligible: false,
      name: 'Секретар Тест',
      photo: null,
      position: 'Секретар',
      profile: null,
      title: null,
    };
    setupFetchMock(undefined, [...PROFESSORS, ineligible]);
    const user = userEvent.setup();
    render(<CommitteePlanner />);

    await waitForProfessors();
    await user.click(screen.getByLabelText(MENTOR_GIVEN_LABEL));
    await user.click(screen.getByTestId(TESTID_MENTOR_SELECT));

    expect(
      screen.getByRole('option', { name: PROFESSOR_NAME_PATTERN }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole('option', { name: SECRETARY_NAME_PATTERN }),
    ).not.toBeInTheDocument();
  });

  it('shows only true alternatives and does not repeat the selected committee', async () => {
    const response = createSuccessResponse();
    response.alternatives.push({
      members: [MENTOR_BOJAN, MENTOR_VLADIMIR],
      mentor: 'Друг Професор',
      score: 0.5,
      supporting_diploma_ids: [],
    });
    setupFetchMock(response);
    const user = userEvent.setup();
    render(<CommitteePlanner />);

    await waitForProfessors();
    await user.type(screen.getByLabelText(TITLE_LABEL), TITLE_SIMPLE);
    await user.click(screen.getByTestId(TESTID_SUBMIT));

    await waitFor(() => {
      expect(screen.getByTestId(TESTID_RESULT)).toBeInTheDocument();
    });

    expect(screen.getByText('Друг Професор')).toBeVisible();
    expect(
      screen.getByText(t('committee.result.alternativeConsider')),
    ).toBeVisible();
    expect(screen.queryByText('Предложена')).not.toBeInTheDocument();
    expect(screen.getAllByText(PROFESSOR_NAME)).toHaveLength(1);
  });
});

const committeeBodies = (fetchMock: Mock<typeof fetch>) =>
  fetchMock.mock.calls
    .filter(([url]) => url === URL_COMMITTEE)
    .map(
      ([, init]) => JSON.parse(init?.body as string) as Record<string, unknown>,
    );

const createShortlistResponse = (): CommitteeResponse => ({
  ...createSuccessResponse(),
  alternatives: [
    {
      members: [MENTOR_BOJAN, MENTOR_VLADIMIR],
      mentor: OTHER_MENTOR,
      score: 0.5,
      supporting_diploma_ids: ['d1'],
    },
  ],
  mentor_is_given: false,
  mode: 'full',
});

describe('CommitteePlanner shortlist and refinement', () => {
  beforeEach(() => {
    vi.stubGlobal('ResizeObserver', ResizeObserverStub);
  });

  afterEach(() => {
    cleanup();
    localStorage.clear();
    vi.unstubAllGlobals();
  });

  it('requests a shortlist of distinct mentors without a chosen mentor', async () => {
    const fetchMock = setupFetchMock(createShortlistResponse());
    const user = userEvent.setup();
    render(<CommitteePlanner />);

    await submitTitle(user);

    expect(committeeBodies(fetchMock)[0]).toMatchObject({
      alternatives: 5,
      distinct_mentors: true,
      mentor_topk: 5,
    });
    expect(screen.getByText(t('committee.result.shortlist'))).toBeVisible();
  });

  it('refines around a mentor chosen from the shortlist', async () => {
    const fetchMock = setupFetchMock(createShortlistResponse());
    const user = userEvent.setup();
    render(<CommitteePlanner />);

    await submitTitle(user);
    await user.click(
      screen.getByRole('button', {
        name: `${t('committee.action.chooseMentor')}: ${OTHER_MENTOR}`,
      }),
    );

    await waitFor(() => {
      expect(committeeBodies(fetchMock)).toHaveLength(2);
    });

    const refined = committeeBodies(fetchMock)[1];

    expect(refined).toMatchObject({ alternatives: 3, mentor: OTHER_MENTOR });
    expect(refined).not.toHaveProperty('distinct_mentors');
  });

  it('excludes a person and lets the student undo it', async () => {
    const fetchMock = setupFetchMock(createShortlistResponse());
    const user = userEvent.setup();
    render(<CommitteePlanner />);

    await submitTitle(user);
    await user.click(
      screen.getByRole('button', {
        name: `${t('committee.action.exclude')}: ${MENTOR_BOJAN}`,
      }),
    );

    await waitFor(() => {
      expect(committeeBodies(fetchMock)[1]).toMatchObject({
        exclude_professors: [MENTOR_BOJAN],
      });
    });

    await user.click(
      await screen.findByRole('button', {
        name: `${t('committee.refine.remove')}: ${MENTOR_BOJAN}`,
      }),
    );

    await waitFor(() => {
      expect(committeeBodies(fetchMock)).toHaveLength(3);
    });

    expect(committeeBodies(fetchMock)[2]).not.toHaveProperty(
      'exclude_professors',
    );
  });

  it('keeps a pinned member in the next request', async () => {
    const fetchMock = setupFetchMock(createShortlistResponse());
    const user = userEvent.setup();
    render(<CommitteePlanner />);

    await submitTitle(user);
    await user.click(
      screen.getByRole('button', {
        name: `${t('committee.action.keep')}: ${MENTOR_VLADIMIR}`,
      }),
    );

    await waitFor(() => {
      expect(committeeBodies(fetchMock)[1]).toMatchObject({
        include_professors: [MENTOR_VLADIMIR],
      });
    });
  });

  it('restores a saved draft and clears it on start over', async () => {
    setupFetchMock();
    const user = userEvent.setup();
    const { unmount } = render(<CommitteePlanner />);

    await waitForProfessors();
    await user.type(screen.getByLabelText(TITLE_LABEL), TITLE_THESIS);
    unmount();
    render(<CommitteePlanner />);
    await waitForProfessors();

    expect(screen.getByLabelText(TITLE_LABEL)).toHaveValue(TITLE_THESIS);

    await user.click(screen.getByTestId(TESTID_SUBMIT));
    await screen.findByTestId(TESTID_RESULT);
    await user.click(
      screen.getByRole('button', { name: t('committee.startOver') }),
    );

    expect(screen.getByLabelText(TITLE_LABEL)).toHaveValue('');
    expect(localStorage).toHaveLength(0);
  });

  it('copies a draft message for the mentor', async () => {
    setupFetchMock(createShortlistResponse());
    const user = userEvent.setup();
    const writeText = vi
      .spyOn(navigator.clipboard, 'writeText')
      .mockResolvedValue();
    render(<CommitteePlanner />);

    await submitTitle(user);
    await user.click(
      screen.getByRole('button', { name: t('committee.copyMessage') }),
    );

    const message = writeText.mock.calls[0]?.[0];

    expect(message).toContain(PROFESSOR_NAME);
    expect(message).toContain(TITLE_SIMPLE);
  });
});

describe('CommitteePlanner interactions', () => {
  beforeEach(() => {
    vi.stubGlobal('ResizeObserver', ResizeObserverStub);
  });

  afterEach(() => {
    cleanup();
    localStorage.clear();
    vi.unstubAllGlobals();
  });

  it('counts only the diplomas that support each person', async () => {
    const response = createSuccessResponse();
    response.mentor.supporting_diploma_ids = ['d1'];
    response.evidence.similar_diplomas.push({
      external_id: '2023-002',
      id: 'd2',
      members: [MENTOR_BOJAN],
      mentor: PROFESSOR_NAME,
      rerank_score: 0.5,
      similarity: 0.5,
      title: 'Друга тема',
    });
    setupFetchMock(response);
    const user = userEvent.setup();
    render(<CommitteePlanner />);

    await submitTitle(user);

    expect(
      screen.getAllByText(`1 ${t('committee.why.defenseOne')}`),
    ).toHaveLength(2);
    expect(screen.getAllByText('Друга тема')).toHaveLength(1);
  });

  it('confirms when the summary is copied', async () => {
    setupFetchMock();
    const user = userEvent.setup();
    const writeText = vi
      .spyOn(navigator.clipboard, 'writeText')
      .mockResolvedValue();
    render(<CommitteePlanner />);

    await submitTitle(user);
    await user.click(
      screen.getByRole('button', { name: t('committee.copySummary') }),
    );

    expect(writeText).toHaveBeenCalledOnce();
    await expect(
      screen.findByText(t('committee.copied')),
    ).resolves.toBeVisible();
  });

  it('moves focus to the title when it is missing', async () => {
    setupFetchMock();
    const user = userEvent.setup();
    render(<CommitteePlanner />);

    await waitForProfessors();
    await user.click(screen.getByTestId(TESTID_SUBMIT));

    expect(screen.getByLabelText(TITLE_LABEL)).toHaveFocus();
  });

  it('filters the mentor picker by search text', async () => {
    setupFetchMock();
    const user = userEvent.setup();
    render(<CommitteePlanner />);

    await waitForProfessors();
    await user.click(screen.getByLabelText(MENTOR_GIVEN_LABEL));
    await user.click(screen.getByTestId(TESTID_MENTOR_SELECT));
    await user.type(
      screen.getByPlaceholderText(t('committee.mentorSearch')),
      'Бојан',
    );

    expect(
      screen.getByRole('option', { name: new RegExp(MENTOR_BOJAN, 'u') }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole('option', { name: PROFESSOR_NAME_PATTERN }),
    ).not.toBeInTheDocument();
  });

  it('groups the mentor mode radios', async () => {
    setupFetchMock();
    render(<CommitteePlanner />);

    await waitForProfessors();

    const radios = screen.getAllByRole('radio');

    expect(radios[0]).toHaveAttribute('name', 'committee-mentor-mode');
    expect(radios[1]).toHaveAttribute('name', 'committee-mentor-mode');
  });
});
